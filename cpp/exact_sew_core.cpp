// Production numerical core. It mirrors the validated geometry implementation
// Eigen, Python callbacks, OpenMP, or fast-math.  Production wiring remains
// intentionally absent until the differential migration phase.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kDegeneracyTolerance = 1e-12;
constexpr double kSp3ExactTolerance = 1e-10;
constexpr double kSp3IntersectionTolerance = 64.0 * std::numeric_limits<double>::epsilon();
constexpr double kSp3ProjectionTolerance = 64.0 * std::numeric_limits<double>::epsilon();
constexpr int kMaximumEventLimit = 1'000'000;

struct Vec3 { double x, y, z; };
struct Mat3 { std::array<double, 9> v{}; };
struct BudgetExceeded : std::runtime_error { BudgetExceeded() : std::runtime_error("event evaluation budget exhausted") {} };
thread_local int* g_alignment_budget_remaining = nullptr;

double stable_norm(const Vec3& v) {
  if (!std::isfinite(v.x) || !std::isfinite(v.y) || !std::isfinite(v.z))
    throw std::invalid_argument("encountered a non-finite intermediate vector");
  // Mirror CPython 3.11 math.hypot's correctly-rounded vector_norm path.
  // std::hypot differs by one ulp on Windows, enough to move SP3 tangencies.
  constexpr double kSplitter = 134217729.0;
  const std::array<double, 3> values{std::abs(v.x), std::abs(v.y), std::abs(v.z)};
  const double maximum = std::max({values[0], values[1], values[2]});
  if (maximum == 0.0) return 0.0;
  int exponent = 0;
  std::frexp(maximum, &exponent);
  if (exponent < -1023) {
    double csum = 1.0, fraction = 0.0;
    for (double value : values) {
      const double square = (value / maximum) * (value / maximum);
      const double old = csum;
      csum += square;
      fraction += (old - csum) + square;
    }
    return maximum * std::sqrt(csum - 1.0 + fraction);
  }
  const double scale = std::ldexp(1.0, -exponent);
  double csum = 1.0, fraction1 = 0.0, fraction2 = 0.0, fraction3 = 0.0;
  for (double value : values) {
    double scaled = value * scale;
    const double split = scaled * kSplitter;
    const double high = split - (split - scaled);
    const double low = scaled - high;
    double term = high * high;
    double old = csum;
    csum += term;
    fraction1 += (old - csum) + term;
    term = 2.0 * high * low;
    old = csum;
    csum += term;
    fraction2 += (old - csum) + term;
    fraction3 += low * low;
  }
  double root = std::sqrt(csum - 1.0 + (fraction1 + fraction2 + fraction3));
  const double split = root * kSplitter;
  const double high = split - (split - root);
  const double low = root - high;
  double term = -high * high;
  double old = csum;
  csum += term;
  fraction1 += (old - csum) + term;
  term = -2.0 * high * low;
  old = csum;
  csum += term;
  fraction2 += (old - csum) + term;
  term = -low * low;
  old = csum;
  csum += term;
  fraction3 += (old - csum) + term;
  const double correction = csum - 1.0 + (fraction1 + fraction2 + fraction3);
  return (root + correction / (2.0 * root)) / scale;
}
double compensated_sum(std::initializer_list<double> values) {
  // CPython math.fsum's partial-sums algorithm, including its final
  // half-even correction.  SP3 uses this rather than FMA so the phase and
  // near-tangent roots retain geometry.py's rounding semantics.
  std::array<double, 8> partials{};
  std::size_t count = 0;
  for (double value : values) {
    std::size_t write = 0;
    for (std::size_t read = 0; read < count; ++read) {
      double partial = partials[read];
      if (std::abs(value) < std::abs(partial)) std::swap(value, partial);
      volatile double high = value + partial;
      volatile double roundoff = high - value;
      volatile double low = partial - roundoff;
      if (low != 0.0) partials[write++] = low;
      value = high;
    }
    count = write;
    if (value != 0.0) partials[count++] = value;
  }
  if (count == 0) return 0.0;
  volatile double high = partials[--count];
  volatile double low = 0.0;
  while (count > 0) {
    const double value = high;
    const double partial = partials[--count];
    high = value + partial;
    volatile double roundoff = high - value;
    low = partial - roundoff;
    if (low != 0.0) break;
  }
  if (count > 0 && ((low < 0.0 && partials[count - 1] < 0.0) ||
                    (low > 0.0 && partials[count - 1] > 0.0))) {
    const double doubled = low * 2.0;
    const double corrected = high + doubled;
    if (corrected - high == doubled) high = corrected;
  }
  return high;
}
double sp3_dot(const Vec3& a, const Vec3& b) {
  return compensated_sum({a.x * b.x, a.y * b.y, a.z * b.z});
}
double dot(const Vec3& a, const Vec3& b) {
  return (a.x * b.x + a.y * b.y) + a.z * b.z;
}
Vec3 add(const Vec3& a, const Vec3& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 sub(const Vec3& a, const Vec3& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 mul(double s, const Vec3& v) { return {s * v.x, s * v.y, s * v.z}; }
Vec3 divide(const Vec3& v, double s) { return {v.x / s, v.y / s, v.z / s}; }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 checked_vector(const py::handle& object, const char* name) {
  const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
  if (!array || array.ndim() != 1 || array.shape(0) != 3)
    throw std::invalid_argument(std::string(name) + " must have shape (3,)");
  const auto value = array.unchecked<1>();
  Vec3 result{value(0), value(1), value(2)};
  if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z))
    throw std::invalid_argument(std::string(name) + " must contain only finite values");
  return result;
}
double checked_scalar(double value, const char* name) {
  if (!std::isfinite(value)) throw std::invalid_argument(std::string(name) + " must be finite");
  return value;
}
Vec3 unit(const Vec3& value, const char* name) {
  const double norm = stable_norm(value);
  if (norm <= kDegeneracyTolerance) throw std::invalid_argument(std::string(name) + " must be nonzero");
  return divide(value, norm);
}
Vec3 sp3_unit(const Vec3& value, const char* name) {
  const double scale = std::max({std::abs(value.x), std::abs(value.y), std::abs(value.z)});
  if (scale == 0.0) throw std::invalid_argument(std::string(name) + " must be nonzero");
  const Vec3 scaled = divide(value, scale);
  const double norm = stable_norm(scaled);
  if (norm <= kDegeneracyTolerance / scale) throw std::invalid_argument(std::string(name) + " must be nonzero");
  return divide(scaled, norm);
}
Vec3 unit_perpendicular(const Vec3& value, const Vec3& axis, const char* name) {
  const Vec3 projection = sub(value, mul(dot(value, axis), axis));
  const double value_norm = stable_norm(value);
  const double projection_norm = stable_norm(projection);
  if (value_norm <= kDegeneracyTolerance) throw std::invalid_argument(std::string(name) + " must be nonzero");
  if (projection_norm <= kDegeneracyTolerance * value_norm)
    throw std::invalid_argument(std::string(name) + " must not be parallel to its rotation axis");
  return divide(projection, projection_norm);
}
Mat3 rotation(const Vec3& axis, double theta) {
  const Vec3 k = unit(axis, "axis");
  checked_scalar(theta, "theta");
  const double s = std::sin(theta), c = std::cos(theta), d = 1.0 - c;
  // Deliberately retain geometry.rot's K, K @ K, I + s*K + d*K2 order.
  // Algebraic reduction changes SP3's observable extreme-scale residual.
  const Mat3 skew{{{0.0, -k.z, k.y, k.z, 0.0, -k.x, -k.y, k.x, 0.0}}};
  Mat3 square{};
  for (int row = 0; row < 3; ++row) {
    for (int column = 0; column < 3; ++column) {
      double value = 0.0;
      for (int inner = 0; inner < 3; ++inner)
        value += skew.v[3 * row + inner] * skew.v[3 * inner + column];
      square.v[3 * row + column] = value;
    }
  }
  Mat3 result{};
  for (int index = 0; index < 9; ++index) {
    const double identity = index % 4 == 0 ? 1.0 : 0.0;
    result.v[index] = (identity + s * skew.v[index]) + d * square.v[index];
  }
  return result;
}
Vec3 matvec(const Mat3& m, const Vec3& v) {
  return {m.v[0]*v.x + m.v[1]*v.y + m.v[2]*v.z,
          m.v[3]*v.x + m.v[4]*v.y + m.v[5]*v.z,
          m.v[6]*v.x + m.v[7]*v.y + m.v[8]*v.z};
}
py::array_t<double> matrix_array(const Mat3& m) {
  py::array_t<double> out(py::array::ShapeContainer{static_cast<py::ssize_t>(3), static_cast<py::ssize_t>(3)}); auto data = out.mutable_unchecked<2>();
  for (py::ssize_t r = 0; r < 3; ++r) for (py::ssize_t c = 0; c < 3; ++c) data(r,c) = m.v[3*r+c];
  return out;
}
double wrap_angle(double angle) {
  if (!std::isfinite(angle)) throw std::invalid_argument("angle must be finite");
  double wrapped = std::fmod(angle + kPi, 2.0 * kPi);
  if (wrapped < 0.0) wrapped += 2.0 * kPi;
  wrapped -= kPi;
  return wrapped == 0.0 ? 0.0 : wrapped;
}
double sp1_impl(const Vec3& p1, const Vec3& p2, const Vec3& k) {
  const Vec3 axis = unit(k, "k");
  const Vec3 first = unit_perpendicular(p1, axis, "p1");
  const Vec3 second = unit_perpendicular(p2, axis, "p2");
  double theta = 2.0 * std::atan2(stable_norm(sub(first, second)), stable_norm(add(first, second)));
  if (dot(axis, cross(first, second)) < 0.0) theta = -theta;
  return theta;
}
std::vector<double> sp4_impl(const Vec3& p, const Vec3& h, const Vec3& k, double d) {
  const Vec3 vector = p, axis = unit(k, "k"), normal = unit(h, "h");
  checked_scalar(d, "d");
  unit_perpendicular(vector, axis, "p"); unit_perpendicular(normal, axis, "h");
  const Vec3 skew_p = cross(axis, vector);
  const Vec3 negative_skew_square_p = mul(-1.0, cross(axis, skew_p));
  const double a0 = dot(normal, skew_p), a1 = dot(normal, negative_skew_square_p);
  const double b = d - dot(normal, axis) * dot(axis, vector);
  const double norm_a_squared = a0*a0 + a1*a1;
  if (norm_a_squared <= std::pow(kDegeneracyTolerance * stable_norm(vector), 2.0))
    throw std::invalid_argument("SP4 is degenerate because the circle-plane equation has no angle dependence");
  const double x0 = a0*b, x1 = a1*b;
  if (norm_a_squared > b*b) {
    const double z = std::sqrt(norm_a_squared - b*b);
    return {std::atan2(x0 + z*a1, x1 - z*a0), std::atan2(x0 - z*a1, x1 + z*a0)};
  }
  return {std::atan2(x0, x1)};
}
std::vector<std::pair<double, double>> sp2_pairs(const Vec3& p1, const Vec3& p2, const Vec3& k1, const Vec3& k2);
py::array_t<double> sp2_impl(const Vec3& p1, const Vec3& p2, const Vec3& k1, const Vec3& k2) {
  const auto pairs = sp2_pairs(p1, p2, k1, k2);
  py::array_t<double> out(py::array::ShapeContainer{static_cast<py::ssize_t>(pairs.size()), static_cast<py::ssize_t>(2)}); auto a = out.mutable_unchecked<2>();
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(pairs.size()); ++i) { a(i,0)=pairs[i].first; a(i,1)=pairs[i].second; }
  return out;
}
std::vector<std::pair<double, double>> sp2_pairs(const Vec3& p1, const Vec3& p2, const Vec3& k1, const Vec3& k2) {
  const Vec3 axis1 = unit(k1, "k1"), axis2 = unit(k2, "k2");
  if (stable_norm(cross(axis1, axis2)) <= kDegeneracyTolerance)
    throw std::invalid_argument("k1 and k2 must not be parallel");
  const Vec3 vector1 = unit(p1, "p1"), vector2 = unit(p2, "p2");
  unit_perpendicular(vector1, axis1, "p1"); unit_perpendicular(vector2, axis2, "p2");
  auto theta1 = sp4_impl(vector1, axis2, axis1, dot(axis2, vector2));
  auto theta2 = sp4_impl(vector2, axis1, axis2, dot(axis1, vector1));
  if (theta1.size() > 1 || theta2.size() > 1) {
    theta1 = {theta1.front(), theta1.back()}; theta2 = {theta2.back(), theta2.front()};
  }
  std::vector<std::pair<double, double>> result;
  for (std::size_t i = 0; i < theta1.size(); ++i) result.emplace_back(theta1[i], theta2[i]);
  return result;
}
double sp3_residual(const Vec3& p1, const Vec3& p2, const Vec3& axis, double distance, double angle, double length_scale) {
  const double normalized = std::abs(stable_norm(sub(matvec(rotation(axis, angle), p1), p2)) - distance);
  const double residual = normalized * length_scale;
  if (!std::isfinite(residual)) throw std::invalid_argument("SP3 residual is outside the representable floating-point range");
  return residual;
}
struct Sp3Native { std::vector<double> angles; std::vector<double> residuals; std::vector<bool> exact; bool degenerate; std::string message; };
Sp3Native sp3_native(const Vec3& first, const Vec3& second, const Vec3& k, double distance, double tolerance) {
  const Vec3 axis = sp3_unit(k, "k");
  checked_scalar(distance, "d"); checked_scalar(tolerance, "exact_tolerance");
  if (distance < 0.0) throw std::invalid_argument("d must be nonnegative");
  if (tolerance < 0.0) throw std::invalid_argument("exact_tolerance must be nonnegative");
  const double length_scale = std::max({1.0, std::abs(first.x), std::abs(first.y), std::abs(first.z), std::abs(second.x), std::abs(second.y), std::abs(second.z), distance});
  const Vec3 p1 = divide(first, length_scale), p2 = divide(second, length_scale);
  const double d = distance / length_scale, norm1=stable_norm(p1), norm2=stable_norm(p2);
  const double coordinate1=sp3_dot(axis,p1), coordinate2=sp3_dot(axis,p2);
  const Vec3 perp1=sub(p1,mul(coordinate1,axis)), perp2=sub(p2,mul(coordinate2,axis));
  const double radius1=stable_norm(perp1), radius2=stable_norm(perp2), axial=coordinate1-coordinate2;
  double effective_scale=std::max({radius1,radius2,std::abs(axial),d}); if (effective_scale==0.0) effective_scale=1.0;
  const double normalized_tolerance=tolerance*std::max(1.0/length_scale,effective_scale);
  bool degenerate=false, force_inexact=false; std::string message; std::vector<double> candidates;
  if (radius1 <= kSp3ProjectionTolerance*norm1 || radius2 <= kSp3ProjectionTolerance*norm2) {
    degenerate=true; candidates={0.0};
    const double residual=sp3_residual(p1,p2,axis,d,0.0,length_scale);
    message = residual/length_scale <= normalized_tolerance ? "perpendicular radius is numerically zero; rotation is underdetermined" : "perpendicular radius is numerically zero; returned constant-distance least-squares representative";
  } else {
    const Vec3 unit1=divide(perp1, radius1), unit2=divide(perp2, radius2);
    const double alpha=sp3_dot(unit1,unit2), beta=sp3_dot(cross(axis,unit1),unit2);
    const double r1=radius1/effective_scale,r2=radius2/effective_scale,a=axial/effective_scale,sd=d/effective_scale;
    const double target=0.5*compensated_sum({r1*r1, r2*r2, a*a, -(sd*sd)});
    const double phase=std::atan2(beta,alpha), normalized_target=target/r1/r2;
    if (normalized_target > 1.0 + kSp3IntersectionTolerance) { candidates={phase}; force_inexact=true; message="no circle-sphere intersection; returned least-squares extremum"; }
    else if (normalized_target < -1.0 - kSp3IntersectionTolerance) { candidates={phase+kPi}; force_inexact=true; message="no circle-sphere intersection; returned least-squares extremum"; }
    else { const double delta=std::acos(std::clamp(normalized_target,-1.0,1.0)); candidates={phase-delta,phase+delta}; }
  }
  for (double& value : candidates) value=wrap_angle(value);
  std::sort(candidates.begin(),candidates.end());
  std::vector<double> angles;
  for (double value : candidates) {
    if (angles.empty() || std::min(std::abs(value-angles.back()),2.0*kPi-std::abs(value-angles.back())) > kDegeneracyTolerance) angles.push_back(value);
  }
  if (angles.size()>1 && std::min(std::abs(angles.front()-angles.back()),2.0*kPi-std::abs(angles.front()-angles.back())) <= kDegeneracyTolerance) angles.pop_back();
  std::vector<double> residuals; std::vector<bool> exact;
  for (double angle : angles) { const double residual=sp3_residual(p1,p2,axis,d,angle,length_scale); residuals.push_back(residual); exact.push_back(residual/length_scale <= normalized_tolerance && !force_inexact); }
  return {std::move(angles), std::move(residuals), std::move(exact), degenerate, std::move(message)};
}
py::dict sp3_impl(const Vec3& first, const Vec3& second, const Vec3& k, double distance, double tolerance) {
  const auto native=sp3_native(first,second,k,distance,tolerance); py::dict result;
  result["angles"]=native.angles; result["residuals"]=native.residuals; result["is_exact"]=native.exact; result["degenerate"]=native.degenerate; result["message"]=native.message.empty()?py::none():py::cast(native.message); return result;
}
template <typename F> double brent(F function, double left, double right, double tolerance, int maximum_iterations) {
  if (!std::isfinite(tolerance) || tolerance <= 0.0)
    throw std::invalid_argument("root tolerance must be positive and finite");
  if (maximum_iterations < 1)
    throw std::invalid_argument("root maximum_iterations must be at least one");
  double a=left,b=right,fa=function(a),fb=function(b);
  if (!std::isfinite(fa)||!std::isfinite(fb)) throw std::invalid_argument("root endpoints must be finite");
  if (fa==0.0) return a; if (fb==0.0) return b;
  if ((fa < 0.0 && fb < 0.0) || (fa > 0.0 && fb > 0.0))
    throw std::invalid_argument("root endpoints must bracket a finite sign change");
  double c=a,fc=fa,d=b-a,e=d;
  for (int iteration=0;iteration<maximum_iterations;++iteration) {
    if ((fb>0.0&&fc>0.0)||(fb<0.0&&fc<0.0)) { c=a;fc=fa;d=e=b-a; }
    if (std::abs(fc)<std::abs(fb)) { a=b; b=c; c=a; fa=fb; fb=fc; fc=fa; }
    const double tol=2.0*std::numeric_limits<double>::epsilon()*std::abs(b)+0.5*tolerance;
    const double midpoint=0.5*(c-b);
    if (std::abs(midpoint)<=tol||fb==0.0) return b;
    if (std::abs(e)>=tol&&std::abs(fa)>std::abs(fb)) {
      const double s=fb/fa; double p,q;
      if (a==c) { p=2.0*midpoint*s; q=1.0-s; }
      else { q=fa/fc; const double r=fb/fc; p=s*(2.0*midpoint*q*(q-r)-(b-a)*(r-1.0)); q=(q-1.0)*(r-1.0)*(s-1.0); }
      if (p>0.0) q=-q; else p=-p;
      if (2.0*p<std::min(3.0*midpoint*q-std::abs(tol*q),std::abs(e*q))) { e=d;d=p/q; } else { d=midpoint;e=d; }
    } else { d=midpoint;e=d; }
    a=b;fa=fb; b += std::abs(d)>tol?d:(midpoint>0.0?tol:-tol); fb=function(b);
  }
  throw std::runtime_error("bracketed root did not converge");
}

Mat3 identity() { Mat3 result{}; result.v[0]=result.v[4]=result.v[8]=1.0; return result; }
Mat3 matmul(const Mat3& a, const Mat3& b) {
  Mat3 result{};
  for (int row=0; row<3; ++row) for (int column=0; column<3; ++column)
    for (int inner=0; inner<3; ++inner) result.v[3*row+column] += a.v[3*row+inner]*b.v[3*inner+column];
  return result;
}
Mat3 transpose(const Mat3& value) {
  Mat3 result{};
  for (int row=0; row<3; ++row) for (int column=0; column<3; ++column) result.v[3*row+column]=value.v[3*column+row];
  return result;
}
Mat3 checked_matrix(const py::handle& object, const char* name) {
  const auto array=py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
  if (!array || array.ndim()!=2 || array.shape(0)!=3 || array.shape(1)!=3) throw std::invalid_argument(std::string(name)+" must have shape (3, 3)");
  const auto value=array.unchecked<2>(); Mat3 result{};
  for (py::ssize_t row=0; row<3; ++row) for (py::ssize_t column=0; column<3; ++column) { result.v[3*row+column]=value(row,column); if (!std::isfinite(value(row,column))) throw std::invalid_argument(std::string(name)+" must be finite"); }
  return result;
}
std::array<Vec3, 8> checked_columns(const py::handle& object, const char* name) {
  const auto array=py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
  if (!array || array.ndim()!=2 || array.shape(0)!=3 || array.shape(1)!=8) throw std::invalid_argument(std::string(name)+" must have shape (3, 8)");
  const auto value=array.unchecked<2>(); std::array<Vec3,8> result{};
  for (py::ssize_t column=0; column<8; ++column) { result[column]={value(0,column),value(1,column),value(2,column)}; if (!std::isfinite(result[column].x)||!std::isfinite(result[column].y)||!std::isfinite(result[column].z)) throw std::invalid_argument(std::string(name)+" must be finite"); }
  return result;
}
std::array<Vec3, 7> checked_axes(const py::handle& object) {
  const auto array=py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(object);
  if (!array || array.ndim()!=2 || array.shape(0)!=3 || array.shape(1)!=7) throw std::invalid_argument("h must have shape (3, 7)");
  const auto value=array.unchecked<2>(); std::array<Vec3,7> result{};
  for (py::ssize_t column=0; column<7; ++column) { result[column]={value(0,column),value(1,column),value(2,column)}; if (!std::isfinite(result[column].x)||!std::isfinite(result[column].y)||!std::isfinite(result[column].z)) throw std::invalid_argument("h must be finite"); }
  return result;
}
bool valid_vector(const Vec3& left, const Vec3& right, double tolerance=1e-10) {
  const double scale=std::max({1.0,stable_norm(left),stable_norm(right)});
  return stable_norm(sub(left,right)) <= tolerance*scale;
}
struct LocalCandidate { double angle, residual; std::array<double,7> q; };

// Strict fixed-slot geometry used for production continuation and recovery. It
// deliberately returns no candidate for an inexact/degenerate subproblem.
std::optional<std::array<double,7>> local_configuration(
    double angle, int slot, const std::array<Vec3,7>& h, const std::array<Vec3,8>& p,
    const Vec3& p17, const Vec3& plane_normal, const Mat3& rotation_07) {
  if (slot < 0 || slot >= 8) return std::nullopt;
  const int i=slot/4, j=(slot%4)/2, k=slot%2;
  const double p17_length=stable_norm(p17), p3_length=stable_norm(p[3]), p5_length=stable_norm(p[5]);
  if (p17_length <= 1e-14) return std::nullopt;
  const Vec3 pwe=mul(p5_length, matvec(rotation(plane_normal,angle), mul(-1.0/p17_length,p17)));
  const auto q1_result=sp3_native(p[1],add(p17,pwe),h[0],p3_length,kSp3ExactTolerance);
  if (i >= static_cast<int>(q1_result.angles.size()) || !q1_result.exact[i]) return std::nullopt;
  const double q1=q1_result.angles[i]; const Mat3 r10=rotation(h[0],-q1);
  const Vec3 vector23=sub(add(matvec(r10,p17),matvec(r10,pwe)),p[1]);
  std::vector<std::pair<double,double>> pairs23;
  try { pairs23=sp2_pairs(p[3],vector23,h[2],mul(-1.0,h[1])); } catch (const std::invalid_argument&) { return std::nullopt; }
  if (j >= static_cast<int>(pairs23.size())) return std::nullopt;
  const double q3=pairs23[j].first,q2=pairs23[j].second;
  if (!valid_vector(matvec(rotation(h[2],q3),p[3]),matvec(rotation(mul(-1.0,h[1]),q2),vector23))) return std::nullopt;
  const Vec3 vector45=sub(matvec(rotation(h[2],-q3),matvec(rotation(h[1],-q2),sub(matvec(r10,p17),p[1]))),p[3]);
  std::vector<std::pair<double,double>> pairs45;
  try { pairs45=sp2_pairs(p[5],vector45,h[4],mul(-1.0,h[3])); } catch (const std::invalid_argument&) { return std::nullopt; }
  if (k >= static_cast<int>(pairs45.size())) return std::nullopt;
  const double q5=pairs45[k].first,q4=pairs45[k].second;
  if (!valid_vector(matvec(rotation(h[4],q5),p[5]),matvec(rotation(mul(-1.0,h[3]),q4),vector45))) return std::nullopt;
  Mat3 r05=identity(); const std::array<double,5> q12345{q1,q2,q3,q4,q5};
  for (int n=0;n<5;++n) r05=matmul(r05,rotation(h[n],q12345[n]));
  try {
    const Vec3 q6_target=matvec(matmul(transpose(r05),rotation_07),h[6]);
    const Vec3 q7_target=matvec(matmul(transpose(rotation_07),r05),h[5]);
    const double q6=sp1_impl(h[6],q6_target,h[5]);
    const double q7=sp1_impl(h[5],q7_target,mul(-1.0,h[6]));
    // The values are independently checked by Python's authoritative
    // pinch-site FK before a result can be accepted.
    return std::array<double,7>{q1,q2,q3,q4,q5,q6,q7};
  } catch (const std::invalid_argument&) { return std::nullopt; }
}

double local_alignment(double angle, int slot, const std::array<Vec3,7>& h, const std::array<Vec3,8>& p,
                       const Vec3& p17, const Vec3& normal, const Mat3& r07) {
  if (g_alignment_budget_remaining != nullptr && (*g_alignment_budget_remaining)-- <= 0) throw BudgetExceeded();
  const auto q=local_configuration(angle,slot,h,p,p17,normal,r07);
  if (!q) return std::numeric_limits<double>::quiet_NaN();
  Mat3 r05=identity(); for (int n=0;n<5;++n) r05=matmul(r05,rotation(h[n],(*q)[n]));
  return dot(h[5],matvec(matmul(transpose(r05),r07),h[6]))-dot(h[5],h[6]);
}
bool final_sp1_valid(const std::array<double, 7>& q, const std::array<Vec3, 7>& h, const Mat3& r07) {
  Mat3 r05=identity(); for (int index=0; index<5; ++index) r05=matmul(r05,rotation(h[index],q[index]));
  const Vec3 q6_target=matvec(matmul(transpose(r05),r07),h[6]);
  const Vec3 q7_target=matvec(matmul(transpose(r07),r05),h[5]);
  return valid_vector(matvec(rotation(h[5],q[5]),h[6]),q6_target,1e-8)
      && valid_vector(matvec(rotation(mul(-1.0,h[6]),q[6]),h[5]),q7_target,1e-8);
}

double sp3_margin(const Vec3& first, const Vec3& second, const Vec3& axis, double distance) {
  const Vec3 direction=unit(axis,"axis"); const double axial=dot(direction,sub(first,second));
  const Vec3 first_perpendicular=sub(first,mul(dot(direction,first),direction));
  const Vec3 second_perpendicular=sub(second,mul(dot(direction,second),direction));
  const double r1=stable_norm(first_perpendicular), r2=stable_norm(second_perpendicular);
  const double c=r1*r1+r2*r2+axial*axial-distance*distance, denominator=2.0*r1*r2;
  if (denominator <= std::numeric_limits<double>::epsilon()*std::max({1.0,r1,r2,std::abs(c)}))
    return std::abs(c)<=1e-12*std::max({1.0,distance*distance,axial*axial}) ? 0.0 : -std::numeric_limits<double>::infinity();
  const double value=c/denominator; return 1.0-value*value;
}
double sp4_margin(const Vec3& vector, const Vec3& normal_value, const Vec3& axis_value, double distance) {
  const Vec3 axis=unit(axis_value,"axis"), normal=unit(normal_value,"normal"); const double vector_norm=stable_norm(vector);
  const Vec3 first=cross(axis,vector), second=mul(-1.0,cross(axis,cross(axis,vector)));
  const double a0=dot(normal,first),a1=dot(normal,second), b=distance-dot(normal,axis)*dot(axis,vector), norm_a=std::hypot(a0,a1);
  if (norm_a<=1e-12*vector_norm) return std::abs(b)<=1e-12*std::max(1.0,vector_norm) ? 0.0 : -std::numeric_limits<double>::infinity();
  const double ratio=b/norm_a; return 1.0-ratio*ratio;
}
double sp2_margin(const Vec3& first, const Vec3& second, const Vec3& axis1_value, const Vec3& axis2_value) {
  const Vec3 axis1=unit(axis1_value,"axis1"),axis2=unit(axis2_value,"axis2"), p1=unit(first,"p1"),p2=unit(second,"p2");
  if (stable_norm(cross(axis1,axis2))<=kDegeneracyTolerance) throw std::invalid_argument("SP2 axes are parallel");
  return std::min(sp4_margin(p1,axis2,axis1,dot(axis2,p2)),sp4_margin(p2,axis1,axis2,dot(axis1,p1)));
}
struct Interval { double left, right; };
struct NativeRoot { double angle; int slot; double residual; std::array<double, 7> q; };
struct EventBudget { int remaining; explicit EventBudget(int maximum) : remaining(maximum) {} void consume() { if (remaining-- <= 0) throw BudgetExceeded(); } };
bool native_root_less(const NativeRoot& left, const NativeRoot& right) {
  if (left.angle != right.angle) return left.angle < right.angle;
  if (left.slot != right.slot) return left.slot < right.slot;
  return left.q < right.q;
}
std::vector<NativeRoot> normalize_native_roots(std::vector<NativeRoot> roots) {
  std::sort(roots.begin(), roots.end(), native_root_less);
  std::vector<NativeRoot> unique;
  for (const auto& root : roots) {
    auto duplicate = std::find_if(unique.begin(), unique.end(), [&](const NativeRoot& old) {
      return old.slot == root.slot && std::abs(old.angle - root.angle) <= 1e-8;
    });
    if (duplicate == unique.end()) {
      unique.push_back(root);
    } else if (std::tie(root.residual, root.angle, root.q)
               < std::tie(duplicate->residual, duplicate->angle, duplicate->q)) {
      *duplicate = root;
    }
  }
  std::sort(unique.begin(), unique.end(), native_root_less);
  return unique;
}
template <typename F> std::pair<double,double> bounded_minimum(F function, double left, double right, double tolerance, int maximum_iterations) {
  // Bounded scalar minimization contract: a
  // deterministic, derivative-free search confined to one partition.
  constexpr double phi=0.3819660112501051; double x=left+phi*(right-left), w=x, v=x;
  double fx=function(x),fw=fx,fv=fx,d=0.0,e=0.0;
  for (int iteration=0;iteration<maximum_iterations;++iteration) {
    const double midpoint=0.5*(left+right), tol1=tolerance*std::abs(x)+1e-15, tol2=2.0*tol1;
    if (std::abs(x-midpoint)<=tol2-0.5*(right-left)) break;
    double p=0.0,q=0.0,r=0.0;
    if (std::abs(e)>tol1) { r=(x-w)*(fx-fv); q=(x-v)*(fx-fw); p=(x-v)*q-(x-w)*r; q=2.0*(q-r); if(q>0)p=-p; q=std::abs(q); const double old=e; e=d; if(std::abs(p)<std::abs(0.5*q*old)&&p>q*(left-x)&&p<q*(right-x)) d=p/q; else { e=x<midpoint?right-x:left-x; d=phi*e; } } else { e=x<midpoint?right-x:left-x; d=phi*e; }
    const double u=x+(std::abs(d)>=tol1?d:(d>0?tol1:-tol1)); const double fu=function(u);
    if(fu<=fx) { if(u<x)right=x;else left=x; v=w;fv=fw;w=x;fw=fx;x=u;fx=fu; } else { if(u<x)left=u;else right=u; if(fu<=fw||w==x){v=w;fv=fw;w=u;fw=fu;}else if(fu<=fv||v==x||v==w){v=u;fv=fu;} }
  }
  return {x,fx};
}
template <typename F> std::vector<Interval> feasible_intervals(F margin, double minimum, double maximum, EventBudget& budget, int partitions=64) {
  constexpr int iterations=24; constexpr double width=1e-12, xtol=1e-12;
  if (partitions < 1) throw std::invalid_argument("event partitions must be positive");
  auto evaluate = [&](double angle) { budget.consume(); return margin(angle); };
  std::vector<Interval> values; std::vector<double> coarse(static_cast<std::size_t>(partitions + 1));
  for(int i=0;i<=partitions;++i) coarse[i]=evaluate(minimum+(maximum-minimum)*i/partitions);
  for(int i=0;i<partitions;++i) { const double left=minimum+(maximum-minimum)*i/partitions,right=minimum+(maximum-minimum)*(i+1)/partitions;
    const double lv=coarse[i],rv=coarse[i+1]; if(!std::isfinite(lv)&&!std::isfinite(rv)) continue;
    double peak=0.5*(left+right),value=evaluate(peak); try { auto result=bounded_minimum([&](double x){return -evaluate(x);},left,right,width,iterations);peak=result.first;value=-result.second; } catch(const BudgetExceeded&) { throw; } catch(...) {continue;}
    if(!std::isfinite(value)||value<0.0)continue; double il=left,ir=right;
    try { if(std::isfinite(lv)&&lv<0.0)il=brent([&](double x){return evaluate(x);},left,peak,xtol,128); if(std::isfinite(rv)&&rv<0.0)ir=brent([&](double x){return evaluate(x);},peak,right,xtol,128); } catch(const BudgetExceeded&) { throw; } catch(...) {continue;}
    if(ir-il>=width||std::abs(value)<=1e-9)values.push_back({il,ir});
  }
  std::sort(values.begin(),values.end(),[](const Interval&a,const Interval&b){return a.left<b.left;}); std::vector<Interval> merged;
  for(const auto& value:values) { if(!merged.empty()&&value.left<=merged.back().right+1e-10)merged.back().right=std::max(merged.back().right,value.right);else merged.push_back(value); } return merged;
}
std::vector<NativeRoot> event_aware_roots_native(const std::array<Vec3,7>& h, const std::array<Vec3,8>& p, const Vec3& p17, const Vec3& normal, const Mat3& r07, double minimum, double maximum, int partitions, int maximum_evaluations) {
  if (partitions < 1 || partitions > kMaximumEventLimit)
    throw std::invalid_argument("event partitions exceed the supported hard maximum");
  if (maximum_evaluations < 1 || maximum_evaluations > kMaximumEventLimit)
    throw std::invalid_argument("maximum_event_evaluations exceeds the supported hard maximum");
  if (partitions >= maximum_evaluations)
    throw std::invalid_argument("event partitions require partitions + 1 <= maximum_event_evaluations");
  const double p17_length=stable_norm(p17),p3_length=stable_norm(p[3]),p5_length=stable_norm(p[5]); std::vector<NativeRoot> output; if(p17_length<=1e-14)return output;
  auto pwe=[&](double angle){return mul(p5_length,matvec(rotation(normal,angle),mul(-1.0/p17_length,p17)));};
  if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum >= maximum) throw std::invalid_argument("event interval must be finite and ordered");
  EventBudget budget(maximum_evaluations);
  struct AlignmentBudgetScope { explicit AlignmentBudgetScope(EventBudget& value) { g_alignment_budget_remaining=&value.remaining; } ~AlignmentBudgetScope() { g_alignment_budget_remaining=nullptr; } } alignment_scope(budget);
  const auto parents=feasible_intervals([&](double angle){return sp3_margin(p[1],add(p17,pwe(angle)),h[0],p3_length);},minimum,maximum,budget,partitions);
  for(int i=0;i<2;++i) for(const auto& parent:parents) {
    std::map<double, std::optional<std::pair<double,Mat3>>> q1_cache;
    auto q1_vector=[&](double angle)->std::optional<std::pair<double,Mat3>> { const auto found=q1_cache.find(angle); if(found!=q1_cache.end())return found->second; const auto result=sp3_native(p[1],add(p17,pwe(angle)),h[0],p3_length,kSp3ExactTolerance); const auto value=(i>=static_cast<int>(result.angles.size())||!result.exact[i]) ? std::optional<std::pair<double,Mat3>>{} : std::optional<std::pair<double,Mat3>>{std::make_pair(result.angles[i],rotation(h[0],-result.angles[i]))}; q1_cache.emplace(angle,value); return value;};
    auto children23=feasible_intervals([&](double angle){try{auto q=q1_vector(angle);if(!q)return -std::numeric_limits<double>::infinity();return sp2_margin(p[3],sub(add(matvec(q->second,p17),matvec(q->second,pwe(angle))),p[1]),h[2],mul(-1.0,h[1]));}catch(const BudgetExceeded&) { throw; } catch(...){return -std::numeric_limits<double>::infinity();}},parent.left,parent.right,budget,partitions);
    for(int j=0;j<2;++j) for(const auto& child:children23) {
      auto leafs=feasible_intervals([&](double angle){try{auto q=q1_vector(angle);if(!q)return -std::numeric_limits<double>::infinity();const Vec3 v23=sub(add(matvec(q->second,p17),matvec(q->second,pwe(angle))),p[1]);auto pairs=sp2_pairs(p[3],v23,h[2],mul(-1.0,h[1]));if(j>=static_cast<int>(pairs.size()))return -std::numeric_limits<double>::infinity();const double q3=pairs[j].first,q2=pairs[j].second;const Vec3 v45=sub(matvec(rotation(h[2],-q3),matvec(rotation(h[1],-q2),sub(matvec(q->second,p17),p[1]))),p[3]);return sp2_margin(p[5],v45,h[4],mul(-1.0,h[3]));}catch(const BudgetExceeded&) { throw; } catch(...){return -std::numeric_limits<double>::infinity();}},child.left,child.right,budget,partitions);
      for (int k = 0; k < 2; ++k) {
        for (const auto& leaf : leafs) {
          const int slot = 4 * i + 2 * j + k;
          constexpr int samples = 5;
          std::array<double, samples> xs{}, ys{};
          for (int n = 0; n < samples; ++n) {
            xs[n] = leaf.left + (leaf.right - leaf.left) * n / (samples - 1);
            ys[n] = local_alignment(xs[n], slot, h, p, p17, normal, r07);
          }
          for (int n = 0; n < samples - 1; ++n) {
            std::vector<double> candidates;
            if (std::isfinite(ys[n]) && std::abs(ys[n]) <= 1e-8) {
              candidates.push_back(xs[n]);
            }
            if (std::isfinite(ys[n]) && std::isfinite(ys[n + 1])
                && ys[n] * ys[n + 1] < 0.0) {
              try {
                candidates.push_back(brent(
                    [&](double x) { return local_alignment(x, slot, h, p, p17, normal, r07); },
                    xs[n], xs[n + 1], 1e-15, 128));
              } catch (const BudgetExceeded&) {
                throw;
              } catch (...) {
              }
            }
            if (candidates.empty()) {
              try {
                const auto tangent = bounded_minimum(
                    [&](double x) { return std::abs(local_alignment(x, slot, h, p, p17, normal, r07)); },
                    xs[n], xs[n + 1], 1e-15, 128);
                if (tangent.second <= 1e-8) candidates.push_back(tangent.first);
              } catch (const BudgetExceeded&) {
                throw;
              } catch (...) {
              }
            }
            for (double angle : candidates) {
              const double residual = std::abs(local_alignment(angle, slot, h, p, p17, normal, r07));
              const auto q = local_configuration(angle, slot, h, p, p17, normal, r07);
              if (!q || !std::isfinite(residual) || residual > 1e-8) continue;
              // Keep the strict terminal SP1 gate immediately before recording
              // a root; local_configuration only derives q6/q7 geometrically.
              if (!final_sp1_valid(*q, h, r07)) continue;
              output.push_back({angle, slot, residual, *q});
            }
          }
        }
      }
    }
  }
  return normalize_native_roots(std::move(output));
}
py::list event_aware_roots(py::object h_object, py::object p_object, py::object p17_object, py::object normal_object, py::object r07_object, double minimum, double maximum, int partitions, int maximum_evaluations) {
  const auto h=checked_axes(h_object); const auto p=checked_columns(p_object,"p"); const Vec3 p17=checked_vector(p17_object,"p17"); const Vec3 normal=checked_vector(normal_object,"plane_normal"); const Mat3 r07=checked_matrix(r07_object,"rotation_07");
  if (partitions < 1 || partitions > kMaximumEventLimit)
    throw std::invalid_argument("event partitions exceed the supported hard maximum");
  if (maximum_evaluations < 1 || maximum_evaluations > kMaximumEventLimit)
    throw std::invalid_argument("maximum_event_evaluations exceeds the supported hard maximum");
  if (partitions >= maximum_evaluations)
    throw std::invalid_argument("event partitions require partitions + 1 <= maximum_event_evaluations");
  std::vector<NativeRoot> roots; { py::gil_scoped_release release; roots=event_aware_roots_native(h,p,p17,normal,r07,minimum,maximum,partitions,maximum_evaluations); }
  py::list output; for(const auto& root:roots) { py::dict item; item["angle"]=root.angle; item["slot"]=root.slot; item["residual"]=root.residual; py::array_t<double> q(7); auto view=q.mutable_unchecked<1>(); for(py::ssize_t i=0;i<7;++i)view(i)=root.q[i]; item["q"]=q; output.append(item); } return output;
}

py::list local_slot_roots(py::object h_object, py::object p_object, py::object p17_object,
                          py::object normal_object, py::object r07_object, int slot,
                          double left, double right, int samples, double root_tolerance) {
  const auto h=checked_axes(h_object); const auto p=checked_columns(p_object,"p");
  const auto p17=checked_vector(p17_object,"p17"), normal=checked_vector(normal_object,"plane_normal");
  const auto r07=checked_matrix(r07_object,"rotation_07");
  if (!std::isfinite(left)||!std::isfinite(right)||right<=left) throw std::invalid_argument("search interval must be finite and ordered");
  if (samples<3) throw std::invalid_argument("samples must be at least three");
  std::vector<std::pair<double,double>> brackets; double previous_angle=left, previous=local_alignment(left,slot,h,p,p17,normal,r07);
  for (int index=1;index<=samples;++index) {
    const double angle=left+(right-left)*static_cast<double>(index)/samples;
    const double value=local_alignment(angle,slot,h,p,p17,normal,r07);
    if (std::isfinite(previous)&&std::isfinite(value) && ((previous<0.0&&value>0.0)||(previous>0.0&&value<0.0))) brackets.emplace_back(previous_angle,angle);
    previous_angle=angle; previous=value;
  }
  py::list output;
  for (const auto& bracket:brackets) {
    const double angle=brent([&](double value){ return local_alignment(value,slot,h,p,p17,normal,r07); },bracket.first,bracket.second,root_tolerance,128);
    const auto q=local_configuration(angle,slot,h,p,p17,normal,r07); if (!q) continue;
    py::dict item; item["angle"]=angle; item["slot"]=slot; item["residual"]=std::abs(local_alignment(angle,slot,h,p,p17,normal,r07));
    py::array_t<double> q_array(7); auto values=q_array.mutable_unchecked<1>(); for (py::ssize_t i=0;i<7;++i) values(i)=(*q)[i]; item["q"]=q_array; output.append(item);
  }
  return output;
}

py::list test_normalize_native_roots(py::list input) {
  std::vector<NativeRoot> roots;
  roots.reserve(input.size());
  for (const py::handle item : input) {
    const py::dict value = py::reinterpret_borrow<py::dict>(item);
    roots.push_back({
        py::cast<double>(value["angle"]), py::cast<int>(value["slot"]),
        py::cast<double>(value["residual"]), py::cast<std::array<double, 7>>(value["q"]),
    });
  }
  const auto normalized = normalize_native_roots(std::move(roots));
  py::list output;
  for (const auto& root : normalized) {
    py::dict value;
    value["angle"] = root.angle;
    value["slot"] = root.slot;
    value["residual"] = root.residual;
    value["q"] = root.q;
    output.append(value);
  }
  return output;
}

bool test_final_sp1_valid(py::object q_object, py::object h_object, py::object r07_object) {
  const auto q_array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(q_object);
  if (!q_array || q_array.ndim() != 1 || q_array.shape(0) != 7)
    throw std::invalid_argument("q must have shape (7,)");
  const auto q_view = q_array.unchecked<1>();
  std::array<double, 7> q{};
  for (py::ssize_t index = 0; index < 7; ++index) {
    q[index] = q_view(index);
    if (!std::isfinite(q[index])) throw std::invalid_argument("q must be finite");
  }
  return final_sp1_valid(q, checked_axes(h_object), checked_matrix(r07_object, "rotation_07"));
}

}  // namespace

PYBIND11_MODULE(_exact_sew_core, module) {
  module.doc() = "Exact-SEW C++ event-search production core.";
  module.def("rot", [](py::object axis, double theta) { return matrix_array(rotation(checked_vector(axis,"axis"),theta)); });
  module.def("sp1", [](py::object p1, py::object p2, py::object axis) { return sp1_impl(checked_vector(p1,"p1"),checked_vector(p2,"p2"),checked_vector(axis,"k")); });
  module.def("sp2", [](py::object p1, py::object p2, py::object k1, py::object k2) { return sp2_impl(checked_vector(p1,"p1"),checked_vector(p2,"p2"),checked_vector(k1,"k1"),checked_vector(k2,"k2")); });
  module.def("sp3", [](py::object p1, py::object p2, py::object axis, double distance, double tolerance) { return sp3_impl(checked_vector(p1,"p1"),checked_vector(p2,"p2"),checked_vector(axis,"k"),distance,tolerance); }, py::arg("p1"),py::arg("p2"),py::arg("axis"),py::arg("distance"),py::arg("exact_tolerance")=kSp3ExactTolerance);
  module.def("_test_brent_quadratic", [](double a, double b, double c, double left, double right, double xtol, int maximum_iterations) { return brent([=](double x){return (a*x+b)*x+c;},left,right,xtol,maximum_iterations); });
  module.def("_test_normalize_native_roots", &test_normalize_native_roots);
  module.def("_test_final_sp1_valid", &test_final_sp1_valid);
  module.def("local_slot_roots", &local_slot_roots, py::arg("h"), py::arg("p"), py::arg("p17"), py::arg("plane_normal"), py::arg("rotation_07"), py::arg("slot"), py::arg("left"), py::arg("right"), py::arg("samples")=65, py::arg("root_tolerance")=1e-12,
             "Find strict sign-bracketed alignment roots for one fixed R-2R-2R-2R slot without Python callbacks.");
  module.def("event_aware_roots", &event_aware_roots, py::arg("h"), py::arg("p"), py::arg("p17"), py::arg("plane_normal"), py::arg("rotation_07"), py::arg("minimum")=-kPi, py::arg("maximum")=0.0, py::arg("partitions")=64, py::arg("maximum_event_evaluations")=50000,
             "Run the strict three-level SP3/SP2/SP2 feasible-interval tree and alignment search in C++.");
}
