# Pipeline flowchart guide

## Edit a pipeline

1. Open the Pipeline tab and select or create a pipeline.
2. Drag an operation from the operation list onto the canvas.
3. Drag ports to connect compatible operations. Drag an existing connection away to disconnect it.
4. Open a node's settings button to edit its parameters.
5. Move nodes to arrange the graph. The canvas supports pan and zoom, and the minimap shows the current viewport.
6. Use the delete button to remove a node. Structural and settings changes save through `/save-pipeline-config/<pipeline_name>`.
7. Restart the backend when the interface marks the saved configuration as requiring it.

The editor restores its viewport when you return to the tab. Undo and redo operate on editor history. Profiling and operation errors appear on the nodes when the backend publishes updates.

## If editing fails

Confirm that a pipeline is selected and that `/get-available-operations` succeeds. A node whose configuration cannot load cannot expose valid ports or settings. Check the browser console and the server log for the failed request. Pipeline operation failures also arrive through the `pipeline_operation_errors` SSE event and highlight affected nodes.
