import os
import re
from groq import Groq
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from flask import Flask, request, jsonify, render_template

# Load .env file for GROQ_API_KEY.
load_dotenv()

# --- Flask App Initialization ---
# Make sure to create a 'templates' folder in the same directory as this script
# and place the 'index.html' file inside it.
app = Flask(__name__)

# --- RepoToMermaidConverter Class (Integrated into the web app) ---
# This is the same class you provided, now integrated to work with Flask.
class RepoToMermaidConverter:
    """
    A class to convert a textual summary of a code repository into a Mermaid.js diagram
    using the Groq API for language model processing.
    """
    DEFAULT_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
    DEFAULT_API_PARAMS = {
        "temperature": 0.2,
        "max_tokens": 2048,
    }

    def __init__(self, model: str = DEFAULT_MODEL, api_params: Optional[Dict[str, Any]] = None):
        """
        Initializes the RepoToMermaidConverter with a Groq client and configuration.
        """
        self.client = Groq()
        self.model = model
        self.api_params = self.DEFAULT_API_PARAMS.copy()
        if api_params:
            self.api_params.update(api_params)

    def _process_with_llm(self, repo_summary: str) -> Optional[str]:
        """
        Processes the repository summary through the configured Groq LLM.
        """
        try:
            prompt = f"""
Analyze the following git repository summary. Your task is to extract the components,
their relationships, and the overall data flow.

Format your entire output for a Mermaid.js flowchart. Follow these rules strictly:
- Represent folders or packages using: `folder: Folder Name` and close them with `end folder`.
- Represent components, files, or modules using: `component: Component Name`.
- Represent relationships or data flow using: `Source Component Name -> Target Component Name`.
- Do NOT include markdown code blocks (```mermaid) in your output.
- Do NOT add any conversational text or explanations outside of the specified format.

Repository Summary:
{repo_summary}
"""
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.api_params['temperature'],
                max_tokens=self.api_params['max_tokens'],
                stream=True
            )

            llm_output = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    llm_output += content
            return llm_output
        except Exception as e:
            print(f"Error processing with Groq: {e}")
            return None

    @staticmethod
    def _sanitize_id(name: str) -> str:
        s = name.strip().replace(' ', '_').replace('-', '_').replace('.', '_')
        return re.sub(r'[^a-zA-Z0-9_]', '', s)

    @staticmethod
    def _sanitize_label(name: str) -> str:
        return re.sub(r'[\[\]"`\n\r]', '', name.strip())

    def generate_mermaid_diagram(self, llm_output: str, diagram_type: str = "flowchart") -> str:
        if diagram_type not in ["graph", "flowchart"]:
            diagram_type = "flowchart"

        mermaid_code = "flowchart TD\n"
        lines = llm_output.split('\n')
        folder_stack = []
        declared_components = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower_line = line.lower()

            if "folder:" in lower_line or "package:" in lower_line:
                folder_name = line.split(':', 1)[1]
                safe_folder_id = self._sanitize_id(f"subgraph_{folder_name}")
                safe_folder_label = self._sanitize_label(folder_name)
                mermaid_code += f"    subgraph {safe_folder_id}[\"{safe_folder_label}\"]\n"
                folder_stack.append(safe_folder_id)
            elif "end folder" in lower_line or "end package" in lower_line:
                if folder_stack:
                    mermaid_code += "    end\n"
                    folder_stack.pop()
            elif "component:" in lower_line or "module:" in lower_line:
                component_name = line.split(':', 1)[1]
                safe_id = self._sanitize_id(component_name)
                if safe_id and safe_id not in declared_components:
                    label = self._sanitize_label(component_name)
                    mermaid_code += f"        {safe_id}[\"{label}\"]\n"
                    declared_components.add(safe_id)
            elif "->" in line:
                parts = line.split("->")
                if len(parts) == 2:
                    source_name, target_name = parts[0].strip(), parts[1].strip()
                    source_id, target_id = self._sanitize_id(source_name), self._sanitize_id(target_name)
                    if source_id and target_id:
                        mermaid_code += f"    {source_id} --> {target_id}\n"
        
        while folder_stack:
            mermaid_code += "    end\n"
            folder_stack.pop()

        return mermaid_code

    def process_repo_summary(self, repo_summary: str, diagram_type: str = "flowchart") -> str:
        llm_output = self._process_with_llm(repo_summary)
        if not llm_output:
            return ""
        return self.generate_mermaid_diagram(llm_output, diagram_type)


# --- Flask Routes ---
@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """API endpoint to generate the Mermaid diagram."""
    if not os.getenv("GROQ_API_KEY"):
        return jsonify({"error": "GROQ_API_KEY not set."}), 500
        
    data = request.json
    repo_summary = data.get('summary')
    diagram_type = data.get('type', 'flowchart')

    if not repo_summary:
        return jsonify({"error": "Repository summary is required."}), 400

    try:
        converter = RepoToMermaidConverter()
        mermaid_code = converter.process_repo_summary(repo_summary, diagram_type)
        if not mermaid_code:
            return jsonify({"error": "Failed to generate diagram. The model may have returned an empty response."}), 500
            
        return jsonify({"mermaid_code": mermaid_code})
    except Exception as e:
        print(f"An error occurred during generation: {e}")
        return jsonify({"error": f"An internal error occurred: {e}"}), 500

# --- Main Execution ---
if __name__ == '__main__':
    # To run this app:
    # 1. Install Flask: pip install Flask
    # 2. Run the script: python app.py
    # 3. Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)
    app.run(debug=True)
