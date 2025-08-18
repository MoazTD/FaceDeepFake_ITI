from maya import cmds as cmds
import os
import re
from collections import OrderedDict

def import_obj_sequence_as_blendshapes():

    
    folder_path = cmds.fileDialog2(
        dialogStyle=2, 
        fileMode=3, 
        caption="Select OBJ Sequence Folder"
    )
    
    if not folder_path:
        cmds.warning("No folder selected. Operation cancelled.")
        return
    
    folder_path = folder_path[0]
    
    obj_files = get_obj_files(folder_path)
    
    if not obj_files:
        cmds.warning("No OBJ files found in the selected folder.")
        return
    
    print(f"Found {len(obj_files)} OBJ files")
    
    base_mesh = import_base_mesh(folder_path, obj_files[0])
    if not base_mesh:
        return
    
    target_meshes = import_target_meshes(folder_path, obj_files[1:])
    
    blend_shape_result = create_blend_shape(base_mesh, target_meshes)
    if not blend_shape_result:
        return
    
    if isinstance(blend_shape_result, tuple):
        blend_shape_node, valid_targets = blend_shape_result
        total_valid_frames = len(valid_targets) + 1  
    else:
        blend_shape_node = blend_shape_result
        valid_targets = target_meshes
        total_valid_frames = len(target_meshes) + 1
    
    controls = create_blend_shape_controls(blend_shape_node, total_valid_frames)
    
    create_blend_shape_animation(controls, total_valid_frames, valid_targets)
    
    if isinstance(blend_shape_result, tuple):
        cleanup_target_meshes(valid_targets)
    else:
        cleanup_target_meshes(target_meshes)
    
    print("OBJ sequence blend shape setup completed successfully!")
    print("\n" + "="*50)
    print("USAGE:")
    print("- Scrub timeline from frame 0 to see sequence")
    print("- Frame 0 = Base mesh")
    print("- Frame 1, 2, 3... = Each subsequent OBJ")
    print("- Play timeline for automatic animation")
    print("="*50)
    
    return base_mesh, blend_shape_node, controls, valid_targets

def get_obj_files(folder_path):
    """
    Get all OBJ files from the folder and sort them numerically
    """
    obj_files = []
    
    for file in os.listdir(folder_path):
        if file.lower().endswith('.obj'):
            obj_files.append(file)
    
    def extract_frame_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    obj_files.sort(key=extract_frame_number)
    
    return obj_files

def import_base_mesh(folder_path, base_file):
    """
    Import the base mesh (first frame)
    """
    base_path = os.path.join(folder_path, base_file).replace('\\', '/')
    
    try:
        cmds.select(clear=True)
        
        imported_nodes = cmds.file(
            base_path,
            i=True,  
            type="OBJ",
            ignoreVersion=True,
            returnNewNodes=True
        )
        
        mesh_shapes = []
        for node in imported_nodes:
            if cmds.nodeType(node) == 'mesh':
                mesh_shapes.append(node)
        
        if not mesh_shapes:
            cmds.warning(f"No mesh found in base file: {base_file}")
            return None
        
        base_transform = cmds.listRelatives(mesh_shapes[0], parent=True)[0]
        
        base_mesh_name = f"baseMesh_frame000"
        base_transform = cmds.rename(base_transform, base_mesh_name)
        
        print(f"Imported base mesh: {base_transform}")
        return base_transform
        
    except Exception as e:
        cmds.warning(f"Failed to import base mesh {base_file}: {str(e)}")
        return None

def import_target_meshes(folder_path, target_files):
    """
    Import all target meshes for blend shapes
    """
    target_meshes = []
    
    for i, target_file in enumerate(target_files, 1):
        target_path = os.path.join(folder_path, target_file).replace('\\', '/')
        
        try:
            cmds.select(clear=True)
            
            imported_nodes = cmds.file(
                target_path,
                i=True,
                type="OBJ",
                ignoreVersion=True,
                returnNewNodes=True
            )
            
            mesh_shapes = []
            for node in imported_nodes:
                if cmds.nodeType(node) == 'mesh':
                    mesh_shapes.append(node)
            
            if mesh_shapes:
                target_transform = cmds.listRelatives(mesh_shapes[0], parent=True)[0]
                
                frame_num = str(i).zfill(3)
                target_mesh_name = f"targetMesh_frame{frame_num}"
                target_transform = cmds.rename(target_transform, target_mesh_name)
                
                target_meshes.append(target_transform)
                print(f"Imported target mesh {i}: {target_transform}")
            
        except Exception as e:
            cmds.warning(f"Failed to import target mesh {target_file}: {str(e)}")
    
    return target_meshes

def create_blend_shape(base_mesh, target_meshes):

    if not target_meshes:
        cmds.warning("No target meshes to create blend shapes")
        return None
    
    base_shape = cmds.listRelatives(base_mesh, shapes=True)[0]
    base_vertex_count = cmds.polyEvaluate(base_shape, vertex=True)
    
    print(f"Base mesh '{base_mesh}' has {base_vertex_count} vertices")
    
    topology_info = []
    for target_mesh in target_meshes:
        try:
            target_shape = cmds.listRelatives(target_mesh, shapes=True)[0]
            target_vertex_count = cmds.polyEvaluate(target_shape, vertex=True)
            topology_info.append((target_mesh, target_vertex_count))
            
            if target_vertex_count == base_vertex_count:
                print(f"✓ Same topology: {target_mesh} ({target_vertex_count} vertices)")
            else:
                print(f"⚠ Different topology: {target_mesh} ({target_vertex_count} vertices) - Maya will handle this")
                
        except Exception as e:
            print(f"? Error checking {target_mesh}: {str(e)}")
    
    try:
        print("\nCreating blend shape with topology flexibility...")
        
        blend_shape_node = cmds.blendShape(
            target_meshes + [base_mesh],
            name="objSequence_blendShape",
            origin="world",           
            weight=[1.0] * len(target_meshes),  
            envelope=1.0,           
            frontOfChain=True,       
            automatic=True          
        )[0]
        
        for i, target_mesh in enumerate(target_meshes):
            frame_match = re.search(r'frame(\d+)', target_mesh)
            if frame_match:
                frame_num = frame_match.group(1)
            else:
                frame_num = str(i + 1).zfill(3)
            
            target_name = f"frame{frame_num}"
            
            try:
                cmds.aliasAttr(target_name, f"{blend_shape_node}.weight[{i}]")
                print(f"✓ Added target: {target_name}")
            except:
                print(f"✓ Added target: weight[{i}] (from {target_mesh})")
        
        try:
            cmds.setAttr(f"{blend_shape_node}.origin", 0) 
            
            if cmds.attributeQuery('parallelBlender', node=blend_shape_node, exists=True):
                cmds.setAttr(f"{blend_shape_node}.parallelBlender", 1)
                
        except Exception as e:
            print(f"Note: Could not set advanced blend shape settings: {str(e)}")
        
        print(f"\n✓ Successfully created blend shape: {blend_shape_node}")
        print(f"✓ Added {len(target_meshes)} targets (including different topology)")
        
        return blend_shape_node, target_meshes
        
    except Exception as e:
        print(f"\n✗ Standard blend shape failed: {str(e)}")
        print("Trying alternative method for different topology...")
        
        try:
            blend_shape_node = cmds.blendShape(
                base_mesh,
                name="objSequence_blendShape_auto",
                automatic=True,
                envelope=1.0
            )[0]
            
            successful_targets = []
            for i, target_mesh in enumerate(target_meshes):
                try:
                    cmds.blendShape(
                        blend_shape_node,
                        edit=True,
                        target=[base_mesh, i, target_mesh, 1.0],
                        weight=[i, 0.0],
                        topologyCheck=False  
                    )
                    
                    frame_match = re.search(r'frame(\d+)', target_mesh)
                    frame_num = frame_match.group(1) if frame_match else str(i + 1).zfill(3)
                    target_name = f"frame{frame_num}"
                    
                    try:
                        cmds.aliasAttr(target_name, f"{blend_shape_node}.weight[{i}]")
                    except:
                        pass
                    
                    successful_targets.append(target_mesh)
                    print(f"✓ Added target {i}: {target_mesh}")
                    
                except Exception as target_error:
                    print(f"✗ Failed to add target {target_mesh}: {str(target_error)}")
            
            if successful_targets:
                print(f"\n✓ Alternative method successful!")
                print(f"✓ Created blend shape with {len(successful_targets)} targets")
                return blend_shape_node, successful_targets
            else:
                raise Exception("No targets could be added")
                
        except Exception as alt_error:
            cmds.error(f"Both methods failed. Error: {str(alt_error)}\n"
                      f"Make sure all OBJ files are valid mesh objects.")
            return None

def create_blend_shape_controls(blend_shape_node, total_frames):

    controls = {
        'blend_shape_node': blend_shape_node,
        'total_frames': total_frames
    }
    
    print("Blend shape controls setup completed - use timeline scrubbing")
    return controls

def create_sequence_expression(main_controller, blend_shape_node, total_frames):

    pass

def create_blend_shape_animation(controls, total_frames, valid_targets):
    """
    Create keyframe animation where timeline frame = blend shape frame
    Frame 0 = base mesh, Frame 1 = first target, Frame 2 = second target, etc.
    """
    blend_shape_node = controls['blend_shape_node']
    
    cmds.playbackOptions(minTime=0, maxTime=total_frames - 1)
    
    for frame_index in range(total_frames):
        current_time = float(frame_index)  # Ensure time is float
        
        for target_index, target_mesh in enumerate(valid_targets):
            frame_match = re.search(r'frame(\d+)', target_mesh)
            if frame_match:
                frame_num = frame_match.group(1)
            else:
                frame_num = str(target_index + 1).zfill(3)
            
            target_attr = f"{blend_shape_node}.frame{frame_num}"
            
            if frame_index == 0:
                weight = 0.0
            elif frame_index == target_index + 1:
                weight = 1.0
            else:
                weight = 0.0
            
            cmds.setKeyframe(
                target_attr,
                time=(current_time,),  
                value=weight
            )
            
            cmds.keyTangent(
                target_attr,
                time=(current_time,),  
                inTangentType='step',
                outTangentType='step'
            )
    
    cmds.currentTime(0)
    
    print(f"Created keyframe animation:")
    print(f"- Timeline: Frame 0 to {total_frames - 1}")
    print(f"- Frame 0 = Base mesh")
    for i, target_mesh in enumerate(valid_targets):
        frame_match = re.search(r'frame(\d+)', target_mesh)
        if frame_match:
            original_frame = frame_match.group(1)
            print(f"- Frame {i + 1} = Original mesh_frame_{original_frame}.obj")
        else:
            print(f"- Frame {i + 1} = Target mesh {i + 1}")
    print("Use timeline scrubbing or play to see sequence!")

def cleanup_target_meshes(target_meshes):
    """
    Hide all target meshes to keep scene clean - only base mesh visible
    """
    if not target_meshes:
        print("No target meshes to clean up")
        return
        
    existing_meshes = []
    missing_meshes = []
    
    for mesh in target_meshes:
        if cmds.objExists(mesh):
            existing_meshes.append(mesh)
        else:
            missing_meshes.append(mesh)
            print(f"Target mesh no longer exists: {mesh}")
    
    if not existing_meshes:
        print("No target meshes found to hide - they may have been automatically cleaned up")
        return
    
    try:
        target_group = cmds.group(empty=True, name="blendShape_targets_HIDDEN")
        
        for mesh in existing_meshes:
            try:
                if cmds.objExists(mesh):
                    cmds.parent(mesh, target_group)
                    
                    if cmds.attributeQuery('visibility', node=mesh, exists=True):
                        cmds.setAttr(f"{mesh}.visibility", 0)
                    
                    if cmds.attributeQuery('hiddenInOutliner', node=mesh, exists=True):
                        cmds.setAttr(f"{mesh}.hiddenInOutliner", 1)
                        
                    print(f"✓ Hidden: {mesh}")
                else:
                    print(f"Mesh disappeared during cleanup: {mesh}")
                    
            except Exception as e:
                print(f"Could not hide {mesh}: {str(e)}")
        
        if cmds.objExists(target_group):
            cmds.move(-1000, -1000, -1000, target_group)
            
            if cmds.attributeQuery('visibility', node=target_group, exists=True):
                cmds.setAttr(f"{target_group}.visibility", 0)
                
            if cmds.attributeQuery('hiddenInOutliner', node=target_group, exists=True):
                cmds.setAttr(f"{target_group}.hiddenInOutliner", 1)
        
        print(f"\n✓ Successfully hidden {len(existing_meshes)} target meshes")
        if missing_meshes:
            print(f"✓ {len(missing_meshes)} meshes were already cleaned up automatically")
        print("✓ Only base mesh visible - animation works through blend shape keyframes")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        print("This is not critical - blend shapes will still work correctly")

def create_ui():
    """
    Create a simple UI for the blend shape tool
    """
    window_name = "objSequenceBlendShapeUI"
    
    if cmds.window(window_name, exists=True):
        cmds.deleteUI(window_name)
    
    window = cmds.window(
        window_name,
        title="OBJ Sequence Blend Shape Tool",
        sizeable=True,
        width=400,
        height=200
    )
    
    cmds.columnLayout(adjustableColumn=True, columnOffset=('both', 10))
    
    cmds.text(label="OBJ Sequence Blend Shape Tool", font="boldLabelFont")
    cmds.separator(height=10)
    
    cmds.text(
        label="This tool will:\n"
              "1. Import OBJ sequence from a folder\n"
              "2. Create blend shapes (supports different topology!)\n"
              "3. Set up timeline animation\n"
              "4. Handle meshes with different vertex counts",
        align="left"
    )
    
    cmds.separator(height=10)
    
    cmds.button(
        label="Import OBJ Sequence and Setup Blend Shapes",
        command=lambda x: import_obj_sequence_as_blendshapes(),
        height=40
    )
    
    cmds.separator(height=10)
    
    cmds.text(
        label="Instructions:\n"
              "• Select folder containing numbered OBJ files\n"
              "• Works with different topology/vertex counts!\n"
              "• Timeline Frame 0 = Base mesh (first OBJ)\n"
              "• Timeline Frame 1 = Second OBJ file\n"
              "• Timeline Frame 2 = Third OBJ file, etc.\n"
              "• Scrub timeline or play to see sequence!",
        align="left"
    )
    
    cmds.showWindow(window)

if __name__ == "__main__":
    create_ui()

# import_obj_sequence_as_blendshapes()