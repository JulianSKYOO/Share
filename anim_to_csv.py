import bpy
import csv
import math
from mathutils import Vector, Euler

# Clears existing objects in teh scene
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

# Importing fbx file
def import_fbx(fbx_path):
    bpy.ops.import_scene.fbx(filepath=fbx_path)

# Retrieve armature object
def get_armature():
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            return obj
    
    return None

# Extracting animation data
def extract_animation_data(armature):
    scene = bpy.context.scene
    frame_start = int(scene.frame_start)
    frame_end = int(scene.frame_end)

    bones = [bone.name for bone in armature.pose.bones]

    animation_data = []
    header = ['frame']

    for bone in bones:
        header.extend([
            f"{bone}_pos_x", f"{bone}_pos_y", f"{bone}_pos_z",
            f"{bone}_rot_x", f"{bone}_rot_y", f"{bone}_rot_z"
        ])

    animation_data.append(header)

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        frame_data = [frame]

        for bone in armature.pose.bones:
            pos = armature.matrix_world @ bone.matrix @ Vector((0, 0, 0))
            rot = (armature.matrix_world @ bone.matrix).to_euler()

            frame_data.extend([
                pos.x, pos.y, pos.z,
                math.degrees(rot.x), math.degrees(rot.y), math.degrees(rot.z)
            ])

        animation_data.append(frame_data)

    return animation_data

# To csv
def save_csv(data, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    
def main():
    fbx_path = "" # fbx path
    output_path = "" # output path

    clear_scene()
    
    import_fbx(fbx_path)

    arm = get_armature()
    if not arm:
        print("no arm in the scene")
        return
    
    animation_data = extract_animation_data(arm)
    save_csv(animation_data, output_path)
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    main()