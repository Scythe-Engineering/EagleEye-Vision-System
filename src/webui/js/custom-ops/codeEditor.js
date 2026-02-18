import { EditorState } from "@codemirror/state";
import { EditorView, basicSetup } from "codemirror";
import { python } from "@codemirror/lang-python";
import { json } from "@codemirror/lang-json";
import { lintGutter, setDiagnostics } from "@codemirror/lint";
import { oneDark } from "@codemirror/theme-one-dark";

function buildExtensions(lang, onChange) {
    let langExtension;
    if (lang === "python") {
        langExtension = python();
    } else if (lang === "json") {
        langExtension = json();
    } else {
        console.warn(`codeEditor: unrecognized language "${lang}", falling back to JSON mode`);
        langExtension = json();
    }
    return [
        basicSetup,
        oneDark,
        langExtension,
        lintGutter(),
        EditorView.updateListener.of((update) => {
            if (update.docChanged && onChange) {
                onChange(update.view.state.doc.toString());
            }
        }),
    ];
}

export class CodeEditor {
    constructor(mountEl, onChange) {
        this._mountEl = mountEl;
        this._onChange = onChange;
        this._lang = "python";
        this._view = null;
        this._init();
    }

    _init() {
        const state = EditorState.create({
            doc: "",
            extensions: buildExtensions(this._lang, this._onChange),
        });
        this._view = new EditorView({ state, parent: this._mountEl });
        this._view.dom.style.height = "100%";
        this._view.dom.style.fontSize = "13px";
        this._mountEl.style.overflow = "hidden";
    }

    setContent(text) {
        const view = this._view;
        view.dispatch({
            changes: { from: 0, to: view.state.doc.length, insert: text },
        });
    }

    getContent() {
        return this._view.state.doc.toString();
    }

    setLanguage(lang) {
        if (lang === this._lang) return;
        this._lang = lang;
        const content = this.getContent();
        const state = EditorState.create({
            doc: content,
            extensions: buildExtensions(lang, this._onChange),
        });
        this._view.setState(state);
    }

    setDiagnostics(diagnostics) {
        const doc = this._view.state.doc;
        const cmDiags = diagnostics
            .map((d) => {
                const line = Math.max(1, Math.min(d.line || 1, doc.lines));
                const lineObj = doc.line(line);
                const from = lineObj.from + Math.max(0, (d.column || 1) - 1);
                const to = Math.min(from + 1, lineObj.to);
                return {
                    from,
                    to,
                    severity: d.severity === "error" ? "error" : "warning",
                    message: `[${d.tool}] ${d.message}`,
                    source: d.tool,
                };
            })
            .filter((d) => d.from <= d.to);

        this._view.dispatch(setDiagnostics(this._view.state, cmDiags));
    }

    clearDiagnostics() {
        this._view.dispatch(setDiagnostics(this._view.state, []));
    }

    destroy() {
        if (this._view) {
            this._view.destroy();
            this._view = null;
        }
    }
}
