import os
import ast
import re
from pydoc_markdown import PydocMarkdown
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer


# FOR LAZYDOCS USE:
# find braindance -name "*.py" ! -name "__init__.py" -type f | tee /dev/tty | xargs -I {} sh -c 'echo "Processing {}"; lazydocs --output-format mdx {}'

def get_local_objects(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())

    local_objects = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            local_objects.append(node.name)
    return local_objects

def clean_and_format_markdown(content, module_name):
    # Start with the module name as the main header
    formatted_content = f"# {module_name}\n\n"
    
    # # Remove any lines starting with '#' (existing headers)
    # content = re.sub(r'^#.*\n', '', content, flags=re.MULTILINE)
    
    # Split the content into sections (classes and functions)
    sections = re.split(r'\n(?=\S)', content)
    
    for section in sections:
        if section.strip():
            # Determine if it's a class or function
            if section.startswith('class '):
                header = section.split('(')[0].replace('class ', '')
                formatted_content += f"## {header}\n\n"
            elif '(' in section:
                header = section.split('(')[0].strip()
                formatted_content += f"### `{header}`\n\n"
            
            # Format parameters
            params = re.findall(r':param (\w+):\s*(.*)', section)
            if params:
                formatted_content += "**Parameters:**\n\n"
                for param, desc in params:
                    formatted_content += f"- `{param}`: {desc}\n"
                formatted_content += "\n"
            
            # Format return value
            returns = re.search(r':return:\s*(.*)', section)
            if returns:
                formatted_content += f"**Returns:** {returns.group(1)}\n\n"
            
            # Add any remaining description
            desc = re.sub(r'(:param .*\n)|(:return:.*\n)', '', section)
            desc = desc.strip()
            if desc:
                formatted_content += f"{desc}\n\n"
    
    return formatted_content

def generate_docs(src_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, src_dir)
                out_file_dir = os.path.join(out_dir, os.path.dirname(rel_path))
                os.makedirs(out_file_dir, exist_ok=True)
                
                module_name = os.path.splitext(file)[0]
                local_objects = get_local_objects(file_path)
                
                if local_objects:
                    session = PydocMarkdown()
                    assert isinstance(session.loaders[0], PythonLoader)
                    session.loaders[0].search_path = [os.path.dirname(file_path)]
                    session.loaders[0].modules = [module_name]
                    
                    assert isinstance(session.renderer, MarkdownRenderer)
                    session.renderer.render_module_header = False
                    session.renderer.render_toc = False
                    
                    modules = session.load_modules()
                    if modules:
                        # Filter the docspec.Module to include only local objects
                        modules[0].members = [m for m in modules[0].members if m.name in local_objects]
                        
                        markdown_content = session.renderer.render_to_string(modules)
                        formatted_content = clean_and_format_markdown(markdown_content, module_name)
                        
                        if formatted_content.strip():
                            out_file_path = os.path.join(out_file_dir, module_name + '.md')
                            with open(out_file_path, 'w') as f:
                                f.write(formatted_content)
                            print(f"Generated documentation for {rel_path}")
                        else:
                            print(f"Skipped {rel_path} (no content after formatting)")
                    else:
                        print(f"Skipped {rel_path} (no module loaded)")
                else:
                    print(f"Skipped {rel_path} (no local objects found)")

if __name__ == '__main__':
    src_dir = 'braindance'
    out_dir = 'docs'
    generate_docs(src_dir, out_dir)
    print("Documentation generation complete.")