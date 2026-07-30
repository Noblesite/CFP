import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "noblesite.Trellis2MLXMeshReport",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Trellis2MLXMeshReport") {
            return;
        }

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const report = Array.isArray(message?.text)
                ? message.text[0]
                : message?.text;
            if (!report) {
                return;
            }

            let widget = this.widgets?.find((candidate) => candidate.name === "mesh_report");
            if (!widget) {
                widget = ComfyWidgets.STRING(
                    this,
                    "mesh_report",
                    ["STRING", { multiline: true }],
                    app,
                ).widget;
                widget.inputEl.readOnly = true;
                widget.inputEl.style.opacity = 0.85;
            }
            widget.value = report;

            const size = this.computeSize();
            size[0] = Math.max(size[0], this.size[0], 470);
            size[1] = Math.max(size[1], this.size[1], 360);
            this.onResize?.(size);
            app.graph.setDirtyCanvas(true, false);
        };
    },
});
