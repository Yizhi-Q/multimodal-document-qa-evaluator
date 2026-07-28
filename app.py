"""Interactive Gradio demo."""

import json
import traceback

import gradio as gr

from mllm_docqa.core import DocumentAnalyzer


DEFAULT_SCHEMA = {
    "vendor": "string or null",
    "document_number": "string or null",
    "date": "YYYY-MM-DD or null",
    "currency": "ISO 4217 code or null",
    "total_amount": "number or null",
}


def analyse(image_path, question, schema_text):
    if not image_path:
        return "Upload a document image first.", None
    try:
        schema = json.loads(schema_text)
        result = DocumentAnalyzer().analyze(image_path, question, schema)
        return "Analysis complete.", result
    except Exception as exc:
        traceback.print_exc()
        return f"Analysis failed: {exc}", None


with gr.Blocks(title="Multimodal Document QA Evaluator") as demo:
    gr.Markdown("# Multimodal Document QA Evaluator\nExtract structured fields and ask grounded questions about a document image.")
    with gr.Row():
        image = gr.Image(type="filepath", label="Document image")
        with gr.Column():
            question = gr.Textbox(value="What is the total amount and who issued this document?", label="Question")
            schema = gr.Code(value=json.dumps(DEFAULT_SCHEMA, indent=2), language="json", label="Field schema")
            button = gr.Button("Analyse document", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    output = gr.JSON(label="Structured result")
    button.click(analyse, [image, question, schema], [status, output])


if __name__ == "__main__":
    demo.launch()

