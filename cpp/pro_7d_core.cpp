#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace py = pybind11;
constexpr double PI = 3.14159265358979323846;
using V = std::array<double, 3>;
using M = std::array<double, 9>;
using Q = std::array<double, 7>;
using J = std::array<std::array<double, 7>, 7>;

static V add(V a, V b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
static V sub(V a, V b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
static V scale(V a, double k) { return {a[0]*k,a[1]*k,a[2]*k}; }
static double dot(V a,V b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
static V cross(V a,V b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
static double norm(V a) { return std::sqrt(dot(a,a)); }
static V unit(V a) {
    const double n=norm(a);
    if (!(n>1e-12) || !std::isfinite(n)) throw std::domain_error("singular Pro SEW geometry");
    return scale(a,1.0/n);
}
static M identity() { return {1,0,0,0,1,0,0,0,1}; }
static V mv(const M& A,V v) { return {A[0]*v[0]+A[1]*v[1]+A[2]*v[2],A[3]*v[0]+A[4]*v[1]+A[5]*v[2],A[6]*v[0]+A[7]*v[1]+A[8]*v[2]}; }
static M mm(const M& A,const M& B) {
    M C{};
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) for(int k=0;k<3;++k) C[3*i+j]+=A[3*i+k]*B[3*k+j];
    return C;
}
static M transpose(const M& A) { return {A[0],A[3],A[6],A[1],A[4],A[7],A[2],A[5],A[8]}; }
static M skew(V v) { return {0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0}; }
static M rodrigues(V axis,double q) {
    const M K=skew(axis), K2=mm(K,K);
    M R=identity();
    for(int i=0;i<9;++i) R[i]+=std::sin(q)*K[i]+(1-std::cos(q))*K2[i];
    return R;
}
static V normalized_derivative(V v,V dv) { return scale(sub(dv,scale(v,dot(v,dv))),1.0); }
static double wrap(double x) { return std::atan2(std::sin(x),std::cos(x)); }

static V so3_log(const M& R) {
    const double cs=std::clamp((R[0]+R[4]+R[8]-1)*0.5,-1.0,1.0);
    const double theta=std::acos(cs);
    V v{R[7]-R[5],R[2]-R[6],R[3]-R[1]};
    if(theta<1e-7) return scale(v,0.5);
    if(PI-theta<1e-5) {
        V a{std::sqrt(std::max(0.0,(R[0]+1)*0.5)),std::sqrt(std::max(0.0,(R[4]+1)*0.5)),std::sqrt(std::max(0.0,(R[8]+1)*0.5))};
        if(a[0]>=a[1] && a[0]>=a[2] && a[0]>1e-8) { a[1]=(R[1]+R[3])/(4*a[0]); a[2]=(R[2]+R[6])/(4*a[0]); }
        else if(a[1]>=a[2] && a[1]>1e-8) { a[0]=(R[1]+R[3])/(4*a[1]); a[2]=(R[5]+R[7])/(4*a[1]); }
        else if(a[2]>1e-8) { a[0]=(R[2]+R[6])/(4*a[2]); a[1]=(R[5]+R[7])/(4*a[2]); }
        return scale(unit(a),theta);
    }
    return scale(v,theta/(2*std::sin(theta)));
}
static M right_jacobian_inverse(V e) {
    const double theta=norm(e); M out=identity();
    const M K=skew(e),K2=mm(K,K);
    const double c=theta<1e-5 ? 1.0/12.0+theta*theta/720.0 :
        1.0/(theta*theta)-(1+std::cos(theta))/(2*theta*std::sin(theta));
    for(int i=0;i<9;++i) out[i]+=0.5*K[i]+c*K2[i];
    return out;
}

static V read_v(const py::array_t<double>& a) {
    if(a.ndim()!=1 || a.shape(0)!=3) throw std::invalid_argument("expected length-3 array");
    auto v=a.unchecked<1>(); return {v(0),v(1),v(2)};
}
static Q read_q(const py::array_t<double>& a) {
    if(a.ndim()!=1 || a.shape(0)!=7) throw std::invalid_argument("expected length-7 q");
    auto v=a.unchecked<1>(); Q q{}; for(int i=0;i<7;++i) q[i]=v(i); return q;
}
static M read_m(const py::array_t<double>& a) {
    if(a.ndim()!=2 || a.shape(0)!=3 || a.shape(1)!=3) throw std::invalid_argument("expected 3x3 rotation");
    auto v=a.unchecked<2>(); M out{}; for(int i=0;i<3;++i) for(int j=0;j<3;++j) out[3*i+j]=v(i,j); return out;
}
static py::array_t<double> array_v(V v) {
    py::array_t<double> a(3); auto o=a.mutable_unchecked<1>(); for(int i=0;i<3;++i)o(i)=v[i]; return a;
}
static py::array_t<double> array_q(Q q) {
    py::array_t<double> a(7); auto o=a.mutable_unchecked<1>(); for(int i=0;i<7;++i)o(i)=q[i]; return a;
}
static py::array_t<double> array_m(M m) {
    py::array_t<double> a({3,3}); auto o=a.mutable_unchecked<2>(); for(int i=0;i<3;++i)for(int j=0;j<3;++j)o(i,j)=m[3*i+j]; return a;
}
static bool linear_solve(std::array<std::array<double,7>,7> A,Q b,Q& x) {
    for(int col=0;col<7;++col) {
        int pivot=col; for(int row=col+1;row<7;++row) if(std::abs(A[row][col])>std::abs(A[pivot][col])) pivot=row;
        if(std::abs(A[pivot][col])<1e-14) return false;
        if(pivot!=col) { std::swap(A[pivot],A[col]); std::swap(b[pivot],b[col]); }
        const double inv=1/A[col][col];
        for(int k=col;k<7;++k) A[col][k]*=inv; b[col]*=inv;
        for(int row=0;row<7;++row) if(row!=col) {
            const double f=A[row][col];
            for(int k=col;k<7;++k) A[row][k]-=f*A[col][k]; b[row]-=f*b[col];
        }
    }
    x=b; return true;
}

class ProCore {
    std::array<V,7> axes0_{},anchors0_{};
    V tool0_{},et_{},er_{};
    M grasp0_{};
    std::array<std::array<double,2>,7> limits_{};

    struct State {
        V p{},s{},e{}; M grasp{}; double psi{};
        std::array<V,7> axes{},anchors{};
        J task_jac{};
    };
    State forward(Q q,bool need_jac=true) const {
        State out{};
        M R=identity(); V t{0,0,0};
        for(int i=0;i<7;++i) {
            out.axes[i]=mv(R,axes0_[i]);
            out.anchors[i]=add(mv(R,anchors0_[i]),t);
            const M Ri=rodrigues(axes0_[i],q[i]);
            t=add(t,mv(R,sub(anchors0_[i],mv(Ri,anchors0_[i]))));
            R=mm(R,Ri);
        }
        out.p=add(mv(R,tool0_),t); out.grasp=mm(R,grasp0_);
        out.s=out.anchors[0]; out.e=out.anchors[3];
        const V sw=sub(out.p,out.s),se=sub(out.e,out.s);
        const double swlen=norm(sw),selen=norm(se);
        const V u=unit(sw),v=unit(se);
        const V raw_n=cross(u,v), n=unit(raw_n);
        const double nlen=norm(raw_n);
        const V raw_ref=cross(sub(u,et_),er_), ref=unit(raw_ref);
        const double reflen=norm(raw_ref);
        const double sn=dot(n,cross(u,ref)),cs=dot(n,ref);
        out.psi=std::atan2(sn,cs);
        if(!need_jac) return out;
        for(int i=0;i<7;++i) {
            const V dp=cross(out.axes[i],sub(out.p,out.anchors[i]));
            const V de=i<4 ? cross(out.axes[i],sub(out.e,out.anchors[i])) : V{0,0,0};
            const V du=scale(sub(dp,scale(u,dot(u,dp))),1/swlen);
            const V dv=scale(sub(de,scale(v,dot(v,de))),1/selen);
            const V dcross=add(cross(du,v),cross(u,dv));
            const V dn=scale(sub(dcross,scale(n,dot(n,dcross))),1/nlen);
            const V dk=cross(du,er_);
            const V dref=scale(sub(dk,scale(ref,dot(ref,dk))),1/reflen);
            const double dsn=dot(dn,cross(u,ref))+dot(n,add(cross(du,ref),cross(u,dref)));
            const double dcs=dot(dn,ref)+dot(n,dref);
            const double dpsi=(cs*dsn-sn*dcs)/(sn*sn+cs*cs);
            for(int j=0;j<3;++j) {
                out.task_jac[j][i]=10*dp[j];
                out.task_jac[3+j][i]=out.axes[i][j];
            }
            out.task_jac[6][i]=dpsi;
        }
        return out;
    }
    struct Residual {
        Q values{}; J jac{}; double cost{};
        double position_error{},orientation_error{},sew_error{};
    };
    Residual residual(Q q,V target_p,M target_r,double target_psi,bool jac=true) const {
        const State state=forward(q,jac);
        Residual out{};
        const V dp=sub(state.p,target_p);
        const V orientation=so3_log(mm(target_r,transpose(state.grasp)));
        out.position_error=norm(dp); out.orientation_error=norm(orientation);
        out.sew_error=std::abs(wrap(state.psi-target_psi));
        for(int k=0;k<3;++k) { out.values[k]=10*dp[k]; out.values[3+k]=orientation[k]; }
        out.values[6]=wrap(state.psi-target_psi);
        for(double x:out.values) out.cost+=0.5*x*x;
        if(jac) {
            const M Jrinv=right_jacobian_inverse(orientation);
            out.jac=state.task_jac;
            for(int i=0;i<7;++i) {
                V axis=state.axes[i]; V col=mv(Jrinv,axis);
                for(int k=0;k<3;++k) out.jac[3+k][i]=-col[k];
            }
        }
        return out;
    }
public:
    ProCore(py::array_t<double> axes,py::array_t<double> anchors,py::array_t<double> tool,
            py::array_t<double> grasp,py::array_t<double> et,py::array_t<double> er,
            py::array_t<double> limits):tool0_(read_v(tool)),et_(read_v(et)),er_(read_v(er)),grasp0_(read_m(grasp)) {
        if(axes.ndim()!=2 || anchors.ndim()!=2 || axes.shape(0)!=7 || axes.shape(1)!=3 || anchors.shape(0)!=7 || anchors.shape(1)!=3 || limits.ndim()!=2 || limits.shape(0)!=7 || limits.shape(1)!=2)
            throw std::invalid_argument("invalid Pro core geometry dimensions");
        auto a=axes.unchecked<2>(),p=anchors.unchecked<2>(),l=limits.unchecked<2>();
        for(int i=0;i<7;++i) {
            for(int k=0;k<3;++k) { axes0_[i][k]=a(i,k); anchors0_[i][k]=p(i,k); }
            axes0_[i]=unit(axes0_[i]);
            limits_[i]={l(i,0),l(i,1)};
        }
    }
    py::dict evaluate(py::array_t<double> q_array) const {
        const Q q=read_q(q_array); const State st=forward(q);
        py::array_t<double> j({7,7}); auto o=j.mutable_unchecked<2>();
        for(int i=0;i<7;++i)for(int k=0;k<7;++k)o(i,k)=st.task_jac[i][k];
        py::dict d;d["position"]=array_v(st.p);d["orientation"]=array_m(st.grasp);d["psi"]=st.psi;d["geometric_jacobian"]=j;
        return d;
    }
    py::dict residual_and_jacobian(py::array_t<double> q_array,py::array_t<double> p_array,
                                   py::array_t<double> r_array,double psi) const {
        const Residual r=residual(read_q(q_array),read_v(p_array),read_m(r_array),psi);
        py::array_t<double> j({7,7});auto o=j.mutable_unchecked<2>();
        for(int i=0;i<7;++i)for(int k=0;k<7;++k)o(i,k)=r.jac[i][k];
        py::dict d;d["residual"]=array_q(r.values);d["jacobian"]=j;
        return d;
    }
    py::array_t<double> predict(py::array_t<double> p_array,py::array_t<double> r_array,
                                double psi,py::array_t<double> previous_array) const {
        Q previous=read_q(previous_array),predicted=previous;
        try {
            const Residual now=residual(previous,read_v(p_array),read_m(r_array),psi);
            Q gradient{},step{};std::array<std::array<double,7>,7> normal{};
            for(int i=0;i<7;++i) {
                for(int row=0;row<7;++row) gradient[i]+=now.jac[row][i]*now.values[row];
                for(int j=0;j<7;++j)for(int row=0;row<7;++row)
                    normal[i][j]+=now.jac[row][i]*now.jac[row][j];
                normal[i][i]+=1e-4;
            }
            for(double& value:gradient)value=-value;
            if(linear_solve(normal,gradient,step)) {
                double max_step=0;for(double value:step)max_step=std::max(max_step,std::abs(value));
                const double factor=std::min(1.0,0.25/std::max(1e-12,max_step));
                for(int i=0;i<7;++i)predicted[i]=std::clamp(
                    previous[i]+factor*step[i],limits_[i][0],limits_[i][1]);
            }
        } catch(const std::domain_error&) {}
        return array_q(predicted);
    }
    py::dict solve(py::array_t<double> p_array,py::array_t<double> r_array,double psi,
                   py::array_t<double> seed_array,int max_iterations,
                   double position_tolerance,double orientation_tolerance,double sew_tolerance) const {
        V p=read_v(p_array);M r=read_m(r_array);Q q=read_q(seed_array);
        for(int i=0;i<7;++i) q[i]=std::clamp(q[i],limits_[i][0],limits_[i][1]);
        double damping=1e-3;int iteration=0;bool success=false;
        for(;iteration<max_iterations;++iteration) {
            Residual now;
            try {now=residual(q,p,r,psi);} catch(const std::domain_error&) {break;}
            if(now.position_error<position_tolerance && now.orientation_error<orientation_tolerance &&
               now.sew_error<sew_tolerance) {success=true;break;}
            Q gradient{};std::array<std::array<double,7>,7> normal{};
            for(int i=0;i<7;++i) {
                for(int row=0;row<7;++row) gradient[i]+=now.jac[row][i]*now.values[row];
                for(int j=0;j<7;++j) for(int row=0;row<7;++row) normal[i][j]+=now.jac[row][i]*now.jac[row][j];
            }
            bool improved=false;
            for(int attempt=0;attempt<12;++attempt) {
                auto A=normal;Q b{},step{};
                for(int i=0;i<7;++i) {
                    b[i]=-gradient[i];A[i][i]+=damping;
                    if((q[i]<=limits_[i][0]+1e-9 && gradient[i]>0) ||
                       (q[i]>=limits_[i][1]-1e-9 && gradient[i]<0)) {
                        for(int j=0;j<7;++j) {A[i][j]=0;A[j][i]=0;} A[i][i]=1;b[i]=0;
                    }
                }
                if(!linear_solve(A,b,step)) {damping*=10;continue;}
                double max_step=0;for(double v:step)max_step=std::max(max_step,std::abs(v));
                const double factor=std::min(1.0,0.5/std::max(1e-12,max_step));
                Q trial=q;for(int i=0;i<7;++i)trial[i]=std::clamp(q[i]+factor*step[i],limits_[i][0],limits_[i][1]);
                try {
                    const Residual candidate=residual(trial,p,r,psi,false);
                    if(candidate.cost<now.cost-1e-12) {q=trial;damping=std::max(1e-9,damping*0.4);improved=true;break;}
                } catch(const std::domain_error&) {}
                damping*=5;
            }
            if(!improved) break;
        }
        py::dict d;d["q"]=array_q(q);d["iterations"]=iteration;d["success"]=success;
        try {const Residual final=residual(q,p,r,psi,false);
             success=success || (final.position_error<position_tolerance &&
                                 final.orientation_error<orientation_tolerance && final.sew_error<sew_tolerance);
             d["position_error_m"]=final.position_error;d["orientation_error_rad"]=final.orientation_error;d["sew_error_rad"]=final.sew_error;d["cost"]=final.cost;}
        catch(const std::domain_error&) {d["cost"]=std::numeric_limits<double>::infinity();}
        return d;
    }
};

PYBIND11_MODULE(_pro_7d_core,m) {
    py::class_<ProCore>(m,"ProCore")
        .def(py::init<py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>,py::array_t<double>>())
        .def("evaluate",&ProCore::evaluate)
        .def("residual_and_jacobian",&ProCore::residual_and_jacobian)
        .def("predict",&ProCore::predict)
        .def("solve",&ProCore::solve);
}
