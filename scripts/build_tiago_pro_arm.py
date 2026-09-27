"""Build the fixed PAL TIAGo Pro right-arm MuJoCo kinematic model.

The pinned PAL Xacro files are authoritative for joints and task frames. PAL's
pal_mjlab XML supplies the visual/inertial gripper subtree and mesh layout.
Run with local checkouts at the commits listed in PROJECT_PATCHES.md.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET


XACRO = "{http://ros.org/wiki/xacro}"
LIMIT_DEG = ((-270, 30), (-140, 65), (-150, 150), (-140, 65), (-90, 210), (-108, 172), (-140, 140))
ARM_MESHES = ["arm_base_link.stl", *[f"arm_{i}_link.stl" for i in range(1, 5)]]
WRIST_MESHES = [f"arm_{i}_link.stl" for i in range(5, 8)]
GRIPPER_MESHES = ["base_link_tc.stl", "inner_finger.stl", "outer_finger.stl", "fingertip.stl"]
SOURCE_REVISIONS = {
    "pal_sea_arm": "78b544fd2d96b8edbc2472e68f653ea5cce15423",
    "pal_pro_gripper": "ed9445716b9534b6d387d88eaf88f0bcb781b5cd",
    "pal_mjlab": "fb3802e53f64dec7946d5a8c1b9bbc2267985887",
}


def _require_revision(path: Path, project: str) -> None:
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    expected = SOURCE_REVISIONS[project]
    if actual != expected:
        raise ValueError(f"{project} checkout must be {expected}, found {actual}")


def _origin_macro(path: Path, name: str) -> tuple[list[float], list[float]]:
    root = ET.parse(path).getroot()
    macro = next((e for e in root.findall(f"{XACRO}macro") if e.get("name") == name), None)
    if macro is None:
        raise ValueError(f"Missing official Xacro macro {name}: {path}")
    origin = macro.find("origin")
    if origin is None:
        raise ValueError(f"Missing origin in {name}: {path}")
    return [float(v) for v in origin.get("xyz").split()], [float(v) for v in origin.get("rpy").split()]


def _quat_from_urdf_rpy(rpy: list[float]) -> str:
    roll, pitch, yaw = (v / 2 for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    q = (cr * cp * cy + sr * sp * sy,
         sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy,
         cr * cp * sy - sr * sp * cy)
    return " ".join(f"{v:.17g}" for v in q)


def _vec(values: list[float]) -> str:
    return " ".join(f"{v:.17g}" for v in values)


def _check_official_sources(arm: Path, gripper: Path) -> None:
    arm_xacro = (arm / "urdf/arm/arm.urdf.xacro").read_text(encoding="utf-8")
    limits = (arm / "urdf/arm/arm_limits/tiago-pro/arm.limits.xacro").read_text(encoding="utf-8")
    wrist_limits = (arm / "urdf/arm/arm_limits/tiago-pro/spherical-wrist.limits.xacro").read_text(encoding="utf-8")
    ee = (arm / "urdf/end_effector/end_effector.urdf.xacro").read_text(encoding="utf-8")
    grip = (gripper / "urdf/gripper.urdf.xacro").read_text(encoding="utf-8")
    for token in ('name="${name}_tool_link"', 'name="${name}_tool_joint"', "arm_type == 'tiago-pro'"):
        if token not in arm_xacro:
            raise ValueError(f"Official arm Xacro changed: {token}")
    for token in ('value="${30 if joint_reflect == 1 else 270}"', 'value="-140"', 'value="150"'):
        if token not in limits:
            raise ValueError(f"Official arm limits changed: {token}")
    for token in ('value="${210 if joint_reflect == 1 else 90}"', 'value="-108"', 'value="140"'):
        if token not in wrist_limits:
            raise ValueError(f"Official wrist limits changed: {token}")
    if '<origin xyz="0.0 0.0 0.0" rpy="0 0 0"/>' not in ee:
        raise ValueError("Official tool-to-gripper mount changed")
    if '<origin xyz="0.0 0.0 0.157157" rpy="0 -1.57 0" />' not in grip:
        raise ValueError("Official Pro grasping frame changed")


def build(arm: Path, gripper: Path, mjlab_xml: Path, output: Path) -> None:
    _require_revision(arm, "pal_sea_arm")
    _require_revision(gripper, "pal_pro_gripper")
    _require_revision(mjlab_xml.parent, "pal_mjlab")
    _check_official_sources(arm, gripper)
    source = ET.parse(mjlab_xml).getroot()
    upstream_right = next(b for b in source.iter("body") if b.get("name") == "arm_right_1_link")
    right = copy.deepcopy(upstream_right)
    model = ET.Element("mujoco", {"model": "tiago_pro_right_arm_only"})
    ET.SubElement(model, "compiler", {"angle": "radian", "meshdir": "meshes", "autolimits": "true", "fusestatic": "false"})
    model.append(copy.deepcopy(source.find("default")))
    assets = ET.SubElement(model, "asset")
    for material in source.find("asset").findall("material"):
        assets.append(copy.deepcopy(material))
    meshes = {
        "arm_base_link": "arm/arm_base_link.stl",
        **{f"arm_{i}_link": f"arm/arm_{i}_link.stl" for i in range(1, 5)},
        **{f"arm_{i}_link": f"arm/spherical-wrist/arm_{i}_link.stl" for i in range(5, 8)},
        **{name.removesuffix('.stl'): f"gripper/{name}" for name in GRIPPER_MESHES},
    }
    for name, file in meshes.items():
        ET.SubElement(assets, "mesh", {"name": name, "file": file})

    world = ET.SubElement(model, "worldbody")
    base = ET.SubElement(world, "body", {"name": "arm_base", "childclass": "tiago_pro"})
    ET.SubElement(base, "site", {"name": "arm_base_site", "size": "0.008"})
    ET.SubElement(base, "geom", {"class": "visual", "mesh": "arm_base_link", "material": "dark_medium_gray"})
    base.append(right)

    arm_props = arm / "urdf/arm/arm_properties/tiago-pro/arm.properties.xacro"
    wrist_props = arm / "urdf/arm/arm_properties/tiago-pro/spherical-wrist.properties.xacro"
    current = right
    for i, (lower, upper) in enumerate(LIMIT_DEG, 1):
        source_file = arm_props if i <= 4 else wrist_props
        pos, rpy = _origin_macro(source_file, f"origin_joint_{i}")
        current.set("pos", _vec(pos))
        current.set("quat", _quat_from_urdf_rpy(rpy))
        joint = current.find("joint")
        if joint is None or joint.get("name") != f"arm_right_{i}_joint":
            raise ValueError(f"Wrong joint in PAL MuJoCo reference at J{i}")
        joint.set("axis", "0 0 -1" if i == 6 else "0 0 1")
        joint.set("range", _vec([math.radians(lower), math.radians(upper)]))
        if i < 7:
            current = next(b for b in current.findall("body") if b.get("name") == f"arm_right_{i + 1}_link")

    j7 = current
    tool_pos, tool_rpy = _origin_macro(wrist_props, "origin_joint_tool_changer")
    # The MJLab reference nests the gripper directly under J7. Reparent it at
    # the exact PAL tool frame, retaining the original fixed finger transforms.
    tool = ET.SubElement(j7, "body", {"name": "arm_right_tool_link", "pos": _vec(tool_pos), "quat": _quat_from_urdf_rpy(tool_rpy)})
    ET.SubElement(tool, "site", {"name": "arm_right_tool_link", "size": "0.008"})
    gripper_base = ET.SubElement(tool, "body", {"name": "gripper_right_base_link"})
    ET.SubElement(gripper_base, "site", {"name": "gripper_right_base_link", "size": "0.008"})
    for geom in list(j7.findall("geom")):
        if geom.get("mesh") == "base_link_tc":
            j7.remove(geom)
            geom.attrib.pop("pos", None)
            gripper_base.append(geom)
    for child in list(j7.findall("body")):
        if child.get("name", "").startswith("gripper_right_"):
            j7.remove(child)
            xyz = [float(v) for v in child.get("pos").split()]
            child.set("pos", _vec([xyz[k] - tool_pos[k] for k in range(3)]))
            gripper_base.append(child)
    # MJLab's ee_right is not the PAL gripper grasping frame.
    for site in list(j7.findall("site")):
        if site.get("name") == "ee_right":
            j7.remove(site)
    ET.SubElement(gripper_base, "site", {
        "name": "gripper_right_grasping_link",
        "pos": "0 0 0.157157",
        "quat": _quat_from_urdf_rpy([0, -1.57, 0]),
        "size": "0.008",
    })
    # Non-task PAL MJLab collision primitives are omitted: this is a kinematic
    # retargeting model, not a collision/dynamics surrogate for the full robot.
    for node in list(model.iter()):
        for geom in list(node.findall("geom")):
            if geom.get("class") == "collision":
                node.remove(geom)
    # The old gripper visual geom has been moved under the official base frame.
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(model, space="  ")
    ET.ElementTree(model).write(output, encoding="utf-8", xml_declaration=True)

    mesh_root = output.parent / "meshes"
    for name in ARM_MESHES:
        src = arm / "meshes/arm_tiago_pro" / name
        dst = mesh_root / "arm" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for name in WRIST_MESHES:
        src = arm / "meshes/arm_tiago_pro/spherical-wrist" / name
        dst = mesh_root / "arm/spherical-wrist" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for name in GRIPPER_MESHES:
        src = gripper / "meshes" / name
        dst = mesh_root / "gripper" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pal-sea-arm", type=Path, required=True, help="pal_sea_arm_description directory at pinned commit")
    parser.add_argument("--pal-pro-gripper", type=Path, required=True, help="pal_pro_gripper_description directory at pinned commit")
    parser.add_argument("--pal-mjlab-xml", type=Path, required=True, help="official PAL MuJoCo reference XML at pinned commit")
    parser.add_argument("--output", type=Path, default=Path("assets/pal_tiago_pro_arm/tiago_pro_arm.xml"))
    args = parser.parse_args()
    build(args.pal_sea_arm, args.pal_pro_gripper, args.pal_mjlab_xml, args.output)
