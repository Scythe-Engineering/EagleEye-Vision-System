# WebUI API reference

The server listens on `0.0.0.0:5001` by default. The API has no authentication. JSON failures generally use `{"message":"..."}` with an appropriate 4xx or 5xx status. Route details and validation live in `src/webui/web_server.py` and its `web_server_utils` mixins.

Path parameters below omit Flask type annotations for readability.

## Page, streams, and status

| Method | Path | Result |
| --- | --- | --- |
| GET | `/`, `/js/main.js`, `/style.css`, `/background.webp` | Built WebUI files. |
| GET | `/assets/<filename>`, `/get-robot-file/<filename>`, `/draco/<filename>` | Asset or decoder file. |
| GET | `/feed/<camera_name>` | Multipart MJPEG camera stream. |
| GET | `/sse/stream` | Named server-sent events for heartbeat, poses, detections, logs, profiling, system status, update progress, compilation progress, and pipeline errors. |
| GET | `/get-system-status` | Current host and NetworkTables status. |
| GET | `/get-log-messages`, `/download-log-file` | Log data or the current log file. |
| GET | `/get_restart_required` | Restart-required flag. |
| POST | `/set_restart_required`, `/restart-backend`, `/shutdown` | Mark restart required, invoke the restart callback, or stop the WebUI server. |

## Pipelines and operations

| Method | Path | Result |
| --- | --- | --- |
| GET | `/get-available-operations` | Operation metadata. |
| GET | `/get-operation-config-data/<operation_name>/<is_secondary>` | One operation schema. |
| POST | `/get-operation-config-data-batch` | Schemas requested in the JSON body. |
| GET | `/get-operation-files/<operation_name>/<parameter_name>` | Managed files for a parameter. |
| POST | `/upload-operation-file/<operation_name>/<parameter_name>` | Upload a parameter file. |
| DELETE | `/delete-operation-file/<operation_name>/<parameter_name>/<filename>` | Delete a parameter file. |
| GET | `/get-pipeline-names` | Configured pipeline names. |
| GET | `/get-pipeline-config/<pipeline_name>` | One pipeline graph. |
| POST | `/save-pipeline-config/<pipeline_name>` | Save one pipeline graph from a JSON body. |
| DELETE | `/delete-pipeline/<pipeline_name>` | Delete one pipeline. |
| GET, PUT | `/pipeline-settings/<pipeline_name>` | Read or replace runtime settings for one pipeline. |
| GET, PUT | `/pipeline-config/json` | Read or replace the complete pipeline configuration. |
| GET | `/get-pipeline-thread-info/<pipeline_name>`, `/get-pipeline-active/<pipeline_name>` | Runtime pipeline state. |

## Visualization and profiling

| Method | Path | Result |
| --- | --- | --- |
| POST | `/start-visualize/<pipeline_name>/<operation_uuid>` | Select an operation for visualization. |
| POST | `/stop-visualize/<pipeline_name>` | Stop visualization. |
| GET | `/visualize/<pipeline_name>` | Current JPEG visualization. |
| GET | `/visualize/stream/<pipeline_name>` | Multipart visualization stream. |
| POST | `/line-profiling/start/<pipeline_name>/<operation_uuid>` | Start line profiling. |
| POST | `/line-profiling/stop/<pipeline_name>/<operation_uuid>` | Stop line profiling. |
| GET | `/line-profiling/status` | Active profiling target. |
| GET | `/line-profiling/report/<pipeline_name>/<operation_uuid>` | Profiling report. |

## Cameras and test video

| Method | Path | Result |
| --- | --- | --- |
| GET | `/get-available-cameras`, `/camera-config/cameras` | Runtime cameras or configured cameras. |
| GET | `/camera-config/<camera_bus_id>` | Camera configuration. |
| POST | `/camera-config/<camera_bus_id>/extrinsics` | Save extrinsics from JSON. |
| POST, DELETE | `/camera-config/<camera_bus_id>/intrinsics` | Upload or delete intrinsics. |
| GET | `/camera-config/<camera_bus_id>/calibration/feed`, `/camera-config/<camera_bus_id>/distortion/feed` | Calibration streams. |
| POST | `/camera-config/<camera_bus_id>/calibration/capture` | Capture a calibration frame. |
| GET | `/camera-config/<camera_bus_id>/calibration/frames` | Captured-frame metadata. |
| GET, DELETE | `/camera-config/<camera_bus_id>/calibration/frames/<frame_index>` | Read or delete a frame. |
| POST | `/camera-config/<camera_bus_id>/calibration/reset`, `/camera-config/<camera_bus_id>/calibration/run` | Reset frames or run calibration. |
| GET, POST | `/test-videos` | List or upload test videos. |
| DELETE | `/test-videos/<filename>` | Delete a test video. |

## Models and visual assets

| Method | Path | Result |
| --- | --- | --- |
| GET | `/device-registry` | Startup inference-device inventory. |
| GET, POST | `/model-library` | List or create model records. |
| PATCH, DELETE | `/model-library/<model_id>` | Update or delete a model record. |
| POST, DELETE | `/model-library/<model_id>/artifacts/<slot>` | Upload or delete an artifact. |
| GET | `/model-library/<model_id>/resolve` | Resolve an artifact for a requested device. |
| GET | `/model-library/mx3-compilation` | Current MX3 compilation state. |
| POST | `/model-library/<model_id>/mx3-compilation` | Start an MX3 compilation job. |
| DELETE | `/model-library/mx3-compilation/<job_id>` | Cancel a job. |
| GET | `/get-available-robots`, `/robot-files` | Robot choices or managed robot records. |
| POST | `/robot-files` | Upload a robot GLB. |
| POST | `/robot-files/<filename>/scale` | Save robot scale. |
| DELETE | `/robot-files/<filename>` | Delete a robot asset. |
| GET, POST | `/field-files` | List or upload field GLBs. |
| POST | `/field-files/<year>/<filename>/scale` | Save field scale. |
| DELETE | `/field-files/<year>/<filename>` | Delete a field asset. |

## Network, settings, and updates

| Method | Path | Result |
| --- | --- | --- |
| GET | `/get-general-conf` | General configuration merged with defaults. |
| POST | `/save-general-conf` | Merge and validate a partial JSON configuration. |
| GET | `/wifi-networks`, `/wifi-networks/status` | Scan results or network-manager status. |
| POST | `/wifi-networks/connect`, `/wifi-networks/disconnect` | Connect using the JSON body or disconnect. |
| GET | `/system-update/status`, `/system-update/info` | Update progress or local and remote git state. |
| POST | `/system-update/run` | Start an update for the optional `branch` in the JSON body. Returns `202` when started. |

Legacy static routes `/frc2025r2.json` and `/src/webui/assets/apriltags/<filename>` remain registered for current frontend references.
