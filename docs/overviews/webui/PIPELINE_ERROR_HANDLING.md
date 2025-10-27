# Pipeline Error Handling for Cameras Without Pipelines

## Overview

The WebUI now gracefully handles cameras that don't have any configured pipelines, integrating seamlessly with the pipeline creation system. This allows users to create pipelines for newly detected cameras without encountering errors.

## Problem Statement

Previously, when a new camera was detected (e.g., `Arducam OV9782 USB Camera`), the WebUI would attempt to fetch pipelines for that camera, resulting in a `KeyError` because the camera wasn't present in `pipeline_config.json`. This would cause the application to crash or display errors.

## Solution

The backend API methods now handle missing cameras and pipelines gracefully, returning appropriate responses that the frontend can work with.

## Backend Changes

### 1. `get_pipeline_names_for_camera(camera_name)`

**Before:**
```python
def get_pipeline_names_for_camera(self, camera_name: str) -> list[str]:
    with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
        return list(json.load(f)[camera_name].keys())  # KeyError if camera doesn't exist!
```

**After:**
```python
def get_pipeline_names_for_camera(self, camera_name: str) -> list[str]:
    with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
        config = json.load(f)
        if camera_name not in config:
            return []  # Gracefully return empty list
        return list(config[camera_name].keys())
```

**Behavior:**
- Returns an empty array `[]` if the camera has no configured pipelines
- Status code: `200` (success, even with empty result)

### 2. `get_pipeline_config(camera_name, pipeline_name)`

**Before:**
```python
def get_pipeline_config(self, camera_name: str, pipeline_name: str) -> dict:
    with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
        return json.load(f)[camera_name][pipeline_name]  # KeyError if missing!
```

**After:**
```python
def get_pipeline_config(self, camera_name: str, pipeline_name: str) -> dict:
    with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
        config = json.load(f)
        if camera_name not in config:
            return []
        if pipeline_name not in config[camera_name]:
            return []
        return config[camera_name][pipeline_name]
```

**Behavior:**
- Returns an empty array `[]` if the camera or pipeline doesn't exist
- Status code: `200` (success, even with empty result)

### 3. `save_pipeline_config(camera_name, pipeline_name)`

**Enhancement:**
```python
def save_pipeline_config(self, camera_name: str, pipeline_name: str) -> tuple[dict, int]:
    with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
        current_config = json.load(f)
        new_data = request.get_json()

        # Create camera entry if it doesn't exist
        if camera_name not in current_config:
            current_config[camera_name] = {}
        
        # Create pipeline entry if it doesn't exist
        if pipeline_name not in current_config[camera_name]:
            current_config[camera_name][pipeline_name] = []
        
        # ... rest of save logic
```

**Behavior:**
- Automatically creates camera entry if it doesn't exist
- Automatically creates pipeline entry if it doesn't exist
- Allows creating new pipelines for any camera
- Only updates backend pipeline objects if they exist

### 4. `delete_pipeline(camera_name, pipeline_name)`

**Enhancement:**
```python
def delete_pipeline(self, camera_name: str, pipeline_name: str) -> tuple[dict, int]:
    with open(os.path.join(src_path, "config", "pipeline_config.json"), "r") as f:
        current_config = json.load(f)
        if camera_name in current_config and pipeline_name in current_config[camera_name]:
            del current_config[camera_name][pipeline_name]
        else:
            return {"message": "Pipeline not found"}, 404
    # ... rest of delete logic
```

**Behavior:**
- Returns `404` if pipeline doesn't exist
- Returns `200` with success message if deleted successfully

## Frontend Integration

The frontend pipeline creator already had error handling for these scenarios:

```javascript
async function fetchPipelinesForCamera(cameraName) {
    try {
        const response = await fetch(
            `${BACKEND_BASE_URL}/get-pipeline-names-for-camera/${encodeURIComponent(cameraName)}`
        );
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const pipelineNames = await response.json();
        
        pipelines = pipelineNames.map(name => ({...}));
    } catch (error) {
        console.error("Failed to fetch pipelines:", error);
        pipelines = [];  // Fallback to empty array
    }
}
```

**Behavior:**
- Handles empty pipeline arrays gracefully
- Shows "Select Pipeline" dropdown even if no pipelines exist
- Enables "New Pipeline" button to create the first pipeline
- Updates UI when pipelines are created

## Workflow for New Cameras

1. **Camera Detection:**
   - New camera detected: `Arducam OV9782 USB Camera`
   - Camera added to available cameras list

2. **Pipeline Fetching:**
   - Frontend calls `/get-pipeline-names-for-camera/Arducam OV9782 USB Camera`
   - Backend returns `[]` (empty array) instead of throwing error
   - Frontend displays empty pipeline dropdown

3. **Pipeline Creation:**
   - User clicks "New Pipeline" button
   - User enters pipeline name (e.g., "detection_pipeline")
   - Frontend creates empty pipeline in memory

4. **Pipeline Saving:**
   - User adds operations to the pipeline
   - Auto-save triggers on every change
   - Backend `/save-pipeline-config` creates:
     - Camera entry in `pipeline_config.json` if it doesn't exist
     - Pipeline entry under that camera if it doesn't exist
     - Operations list for the pipeline

5. **Result:**
   - Pipeline successfully created and saved
   - Configuration file updated:
     ```json
     {
       "basic_test": { ... },
       "Arducam OV9782 USB Camera": {
         "detection_pipeline": [
           { "action_name": "...", "action_params": {} }
         ]
       }
     }
     ```

## Benefits

1. **No Errors:** Cameras without pipelines don't cause application errors
2. **Seamless Creation:** Users can create pipelines for any camera
3. **Graceful Degradation:** Empty states handled properly throughout
4. **Automatic Initialization:** Camera/pipeline entries created on-demand
5. **Better UX:** Clear visual feedback about pipeline status

## Error Handling Summary

| Scenario | Backend Response | Frontend Behavior |
|----------|------------------|-------------------|
| Camera has pipelines | `200` with array of pipeline names | Shows pipeline dropdown with options |
| Camera has no pipelines | `200` with empty array `[]` | Shows empty dropdown, enables "New Pipeline" button |
| Get config for non-existent pipeline | `200` with empty array `[]` | Handles gracefully, shows empty builder |
| Save pipeline for new camera | `200` success, creates camera/pipeline entries | Pipeline saved and appears in dropdown |
| Delete non-existent pipeline | `404` with error message | Shows error to user |
| Delete existing pipeline | `200` success message | Pipeline removed from dropdown |

## Testing Scenarios

### Scenario 1: New Camera Without Pipelines
1. Start system with new USB camera
2. Navigate to Pipeline tab
3. Select the new camera from dropdown
4. Verify: Pipeline dropdown is empty but functional
5. Click "New Pipeline"
6. Create a pipeline with some operations
7. Verify: Pipeline saves successfully

### Scenario 2: Existing Camera With Pipelines
1. Select camera with existing pipelines
2. Verify: Pipelines appear in dropdown
3. Select a pipeline
4. Verify: Pipeline loads correctly
5. Modify pipeline
6. Verify: Changes save successfully

### Scenario 3: Creating Multiple Pipelines
1. Select a camera
2. Create first pipeline
3. Create second pipeline for same camera
4. Verify: Both pipelines appear in dropdown
5. Switch between pipelines
6. Verify: Each pipeline loads correctly

## Configuration File Structure

After creating pipelines for new cameras, the `pipeline_config.json` structure:

```json
{
  "basic_test": {
    "apriltag_pipeline": [
      { "action_name": "camera_adjust", "action_params": {} },
      { "action_name": "detect_apriltags", "action_params": {} }
    ]
  },
  "Arducam OV9782 USB Camera": {
    "detection_pipeline": [
      { "action_name": "camera_adjust", "action_params": {} }
    ],
    "tracking_pipeline": [
      { "action_name": "camera_adjust", "action_params": {} },
      { "action_name": "temporal_acceleration", "action_params": {} }
    ]
  }
}
```

## Future Enhancements

- Add validation for camera names (URL encoding, special characters)
- Provide UI feedback when creating first pipeline for a camera
- Add pipeline templates for common use cases
- Export/import pipeline configurations
- Pipeline versioning and history

