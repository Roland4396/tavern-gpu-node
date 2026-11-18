import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const labels = {
    submitting: "提交", queued: "排队", assigned: "已分配",
    starting_worker: "加载模型", submitted: "准备输入", preparing: "准备输入",
    running: "准备输入", denoising: "采样", decoding: "解码视频",
    publishing: "保存视频", copying: "复制视频", completed: "完成",
    failed: "失败", cancelled: "已取消", cancel_requested: "取消中",
};
const types = new Set(["MiniMaxH3FL2VASampler", "MiniMaxH3Ref2VASampler"]);

function statusWidget(node) {
    if (!node._h3StatusWidget) {
        node._h3StatusWidget = node.addWidget("text", "H3 进度", "等待任务", () => {}, { serialize: false });
        node._h3StatusWidget.options.readOnly = true;
    }
    return node._h3StatusWidget;
}

app.registerExtension({
    name: "gpu-runtime.H3Progress",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!types.has(nodeData.name)) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = original?.apply(this, args);
            statusWidget(this);
            return result;
        };
    },
});

api.addEventListener("h3_progress", ({ detail }) => {
    const node = app.graph?.getNodeById(detail.node);
    if (!node || !types.has(node.comfyClass || node.type)) return;
    statusWidget(node).value = `${labels[detail.phase] || detail.phase} · ${detail.value}/${detail.max} 步`;
    app.graph.setDirtyCanvas(true, false);
});
