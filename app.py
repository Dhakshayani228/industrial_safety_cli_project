import gradio as gr
from scripts.search import search  # ✅ re-use your existing search function

def search_documents(query):
    if not query.strip():
        return "Please enter a query.", []

    results = search(query)  # search() should return a list of tuples: (file, score, snippet)

    if not results:
        return f"No results found for: {query}", []

    output = []
    for file, score, snippet in results:
        text_block = f"**File:** {file}\n**Score:** {score}\n\n{snippet.strip()}"
        output.append([text_block])  # wrapped in list for Dataframe format

    return f"Results for: {query}", output

with gr.Blocks() as demo:
    gr.Markdown("# 🔍 Industrial Safety Document Search")
    gr.Markdown("Enter your query to search across industrial safety manuals and reports.")

    with gr.Row():
        query_input = gr.Textbox(label="Enter your query", placeholder="e.g. What is industrial safety?", lines=1)
        search_button = gr.Button("Search")

    results_title = gr.Label(label="Status")
    results_output = gr.Dataframe(headers=["Search Results"], datatype=["markdown"], interactive=False)

    search_button.click(
        fn=search_documents,
        inputs=query_input,
        outputs=[results_title, results_output]
    )

if __name__ == "__main__":
    demo.launch()

