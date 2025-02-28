import uuid
import asyncio
import os
import sys
import json
import re
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from open_deep_research.graph import builder
from langgraph.types import Command

# Load and set environment variables from .env file
def setup_environment():
    # Load .env file
    if not load_dotenv():
        print("Warning: .env file not found or empty. Creating from .env.example if available...")
        # Try to load from .env.example if .env doesn't exist
        if os.path.exists(".env.example"):
            with open(".env.example", "r") as example_file:
                with open(".env", "w") as env_file:
                    env_file.write(example_file.read())
            print("Created .env file from .env.example. Please edit with your actual API keys.")
            load_dotenv()  # Load the newly created .env file
        else:
            print("No .env.example file found. Please create a .env file with your API keys.")
    
    # Define all required API keys
    required_keys = [
        "OPENAI_API_KEY",
        "TAVILY_API_KEY",
        "LANGCHAIN_API_KEY",
    ]
    
    # Read keys from .env and set them directly in the environment
    missing_keys = []
    for key in required_keys:
        value = os.getenv(key)
        if not value or value.startswith("your_") or value == "":
            missing_keys.append(key)
        else:
            # Explicitly set the environment variable to ensure it's available
            os.environ[key] = value
            # Print confirmation for debugging (uncomment if needed)
            # print(f"Set {key} in environment (length: {len(value)})")
    
    if missing_keys:
        print("ERROR: The following API keys are missing or invalid in your .env file:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set valid API keys in your .env file and run the script again.")
        print("Required keys: " + ", ".join(required_keys))
        return False
    
    # Double-check Tavily API key specifically since it's causing issues
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        print("ERROR: TAVILY_API_KEY environment variable is not set correctly.")
        print("Please ensure it's properly set in your .env file or environment.")
        return False
    else:
        print(f"✅ TAVILY_API_KEY found (length: {len(tavily_key)})")
    
    # Try importing tavily to check if it can see the key
    try:
        import tavily
        print("✅ Tavily package imported successfully")
        
        # Some packages check their own env vars during import, others when creating clients
        # If needed, you might explicitly set the API key for the package
        # For example: tavily.api_key = tavily_key  (if tavily has this attribute)
    except ImportError:
        print("Warning: Tavily package not found. Make sure it's installed.")
    
    print("✅ All environment variables set successfully!")
    return True

def get_user_input(prompt, allow_file=False, multiline=False, allow_combined=False, debug=True):
    """
    Get input from the user with enhanced options:
    - allow_file: Enable file-based input
    - multiline: Enable multi-line input for command line
    - allow_combined: Enable combining direct text and file content
    - debug: Enable detailed debug output
    """
    print(f"\n{prompt}")
    
    if allow_file and allow_combined:
        print("You can:")
        print("1. Type your input directly")
        print("2. Use 'file:path/to/file.txt' to read entirely from a file")
        print("3. Include file content inline with {{file:path/to/file.txt}} (no quotes needed)")
        print("4. Enter text first, then choose to append file content")
    elif allow_file:
        print("You can type your input directly, or use 'file:path/to/file.txt' to read from a file.")
    
    if multiline:
        print("For multi-line input, type your text and end with a line containing only 'EOF'")
        lines = []
        while True:
            line = input()
            if line.strip() == "EOF":
                break
            lines.append(line)
        user_input = "\n".join(lines)
    else:
        user_input = input("> ").strip()
    
    if debug:
        print(f"\nDEBUG: Input received: '{user_input[:50]}...' (length: {len(user_input)})")
    
    # Check if the input is a direct file reference
    if allow_file and user_input.startswith("file:"):
        file_path = user_input[5:].strip()
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                print(f"Successfully read content from '{file_path}' ({len(content)} characters)")
                return content
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return get_user_input(f"Please try again or enter text directly:", allow_file=True)
    
    # Check for embedded file references if combined mode is enabled
    if allow_file and allow_combined:
        if debug:
            print("\nDEBUG: Checking for file references...")
        
        # Find all embedded file references with more focused pattern matching
        patterns = [
            r"\{\{file:(.*?)\}\}",                  # Basic pattern {{file:path.txt}}
            r"'\{\{file:(.*?)\}\}'",                # Single-quoted '{{file:path.txt}}'
            r'"\{\{file:(.*?)\}\}"',                # Double-quoted "{{file:path.txt}}"
            # Removed problematic pattern that was causing incorrect matches
        ]
        
        all_refs = []
        for pattern in patterns:
            if debug:
                print(f"DEBUG: Using pattern: {pattern}")
            file_refs = re.findall(pattern, user_input)
            if file_refs and debug:
                print(f"DEBUG: Found refs with pattern {pattern}: {file_refs}")
            all_refs.extend(file_refs)
        
        # Cleanse file paths - remove any trailing braces or quotes
        clean_refs = []
        for ref in all_refs:
            ref = ref.strip()
            # Clean up any stray characters
            ref = ref.rstrip('}"\'')
            if ref and ref not in clean_refs:
                clean_refs.append(ref)
        
        if clean_refs:
            print(f"Found {len(clean_refs)} file references to process: {clean_refs}")
            
            # Process all found file references
            for file_path in clean_refs:
                file_path = file_path.strip()
                resolved_path = resolve_file_path(file_path)
                
                if not resolved_path:
                    print(f"❌ File not found: '{file_path}'")
                    continue
                else:
                    print(f"📂 Found file: '{file_path}' at {resolved_path}")
                
                # These are the patterns we'll try to replace in the text
                possible_references = [
                    f"{{{{file:{file_path}}}}}",
                    f"'{{{{file:{file_path}}}}}'",
                    f'"{{{{file:{file_path}}}}}"',
                ]
                
                try:
                    with open(resolved_path, 'r', encoding='utf-8') as file:
                        file_content = file.read()
                        print(f"✅ Read file '{file_path}' ({len(file_content)} characters)")
                        
                        # Replace all possible reference formats with the content
                        replacements_made = False
                        for ref in possible_references:
                            if ref in user_input:
                                if debug:
                                    print(f"DEBUG: Replacing '{ref}' with file content")
                                old_input = user_input
                                user_input = user_input.replace(ref, file_content)
                                replacements_made = True
                                
                                if debug and user_input != old_input:
                                    print(f"DEBUG: Replacement successful. New length: {len(user_input)}")
                        
                        if not replacements_made:
                            # Try a more flexible matching approach
                            original_input = user_input
                            # Use regex to do a more flexible replacement
                            pattern = re.escape(f"{{file:{file_path}}}").replace("\\{", "{").replace("\\}", "}")
                            user_input = re.sub(pattern, file_content, user_input)
                            
                            # Check if any replacements were made with the flexible approach
                            if user_input != original_input:
                                print(f"DEBUG: Made replacements using flexible pattern matching")
                            else:
                                print(f"⚠️ Warning: Found file '{file_path}' but couldn't match any reference patterns.")
                                print(f"   The file content isn't being included in your query.")
                except Exception as e:
                    print(f"❌ Error reading file '{file_path}': {str(e)}")
    
    # Offer to append a file after entering text
    if allow_file and allow_combined and not user_input.startswith("file:") and "{{file:" not in user_input:
        append_choice = input("\nWould you like to append content from a file? (yes/no): ").strip().lower()
        if append_choice in ["y", "yes"]:
            file_path = input("Enter the file path: ").strip()
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    file_content = file.read()
                    print(f"Appending content from '{file_path}' ({len(file_content)} characters)")
                    # Add a newline between user input and file content if needed
                    if user_input and not user_input.endswith("\n"):
                        user_input += "\n\n"
                    user_input += file_content
            except Exception as e:
                print(f"Error reading file: {str(e)}")
    
    if debug:
        print(f"\nDEBUG: Final input length: {len(user_input)}")
        print(f"DEBUG: First 100 characters: '{user_input[:100]}...'")
        print(f"DEBUG: Last 100 characters: '...{user_input[-100:]}'")
    
    return user_input

def format_report_plan(event_data):
    """Format the report plan in a more readable way."""
    # Check if this is the report plan data
    if isinstance(event_data, dict) and 'generate_report_plan' in event_data:
        sections = event_data['generate_report_plan'].get('sections', [])
        
        print("\n📋 REPORT PLAN:\n")
        print("=" * 80)
        
        for i, section in enumerate(sections, 1):
            print(f"SECTION {i}: {section.name}")
            print("-" * 80)
            print(f"Description: {section.description}")
            print(f"Research needed: {'Yes' if section.research else 'No'}")
            print("=" * 80)
        
        return True
    
    # Check if this is the interrupt message containing the formatted plan
    elif isinstance(event_data, dict) and '__interrupt__' in event_data:
        interrupt_data = event_data['__interrupt__']
        if hasattr(interrupt_data, 'value') and isinstance(interrupt_data.value, str):
            # Extract the formatted plan from the interrupt message
            plan_text = interrupt_data.value
            
            # Find the plan portion before the "Does the report plan meet your needs?" text
            plan_parts = plan_text.split("Does the report plan meet your needs?")
            if len(plan_parts) > 1:
                formatted_plan = plan_parts[0].strip()
                
                # The plan is already well-formatted in the interrupt, just print it
                print("\n📋 REPORT PLAN:\n")
                print(formatted_plan)
                
                return True
    
    return False

def format_report_content(event_data):
    """Format the final report in a more readable way."""
    # Try to extract final report content
    try:
        if isinstance(event_data, dict):
            # There are a few ways the report might be structured in the event
            
            # Check if it's a section update with content
            if 'update_section' in event_data:
                section_data = event_data['update_section']
                if hasattr(section_data, 'content') and section_data.content:
                    section_name = getattr(section_data, 'name', 'Section')
                    content = section_data.content
                    
                    print(f"\n{'=' * 80}")
                    print(f"📑 {section_name}")
                    print(f"{'-' * 80}\n")
                    print(content)
                    print(f"\n{'=' * 80}")
                    return True
            
            # Check if it's a final report
            if 'final_report' in event_data:
                report = event_data['final_report']
                if report:
                    print(f"\n{'=' * 80}")
                    print(f"📊 FINAL REPORT")
                    print(f"{'-' * 80}\n")
                    print(report)
                    print(f"\n{'=' * 80}")
                    return True
    except Exception:
        # If any error occurs during formatting, we'll fall back to raw output
        pass
    
    return False

def save_report_to_file(report, topic, thread_id):
    """Save the final report to a file."""
    if not report:
        return None
        
    # Create a safe filename from the topic
    safe_topic = re.sub(r'[^\w\s-]', '', topic).strip().replace(' ', '_').lower()
    filename = f"report_{safe_topic}_{thread_id[:8]}.md"
    
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"# Research Report: {topic}\n\n")
            file.write(report)
        
        return filename
    except Exception as e:
        print(f"Error saving report to file: {str(e)}")
        return None

# Add this function to check if files exist and build working file paths
def resolve_file_path(path, base_paths=None):
    """Try to find a file by checking different base paths."""
    # If path is absolute or directly accessible, return it
    if os.path.exists(path):
        return path
    
    # Check with base paths if specified
    if base_paths:
        for base in base_paths:
            full_path = os.path.join(base, path)
            if os.path.exists(full_path):
                return full_path
    
    # If running from project root, check in different folders
    common_dirs = [".", "source_docs", "data", "files", "resources"]
    for directory in common_dirs:
        full_path = os.path.join(directory, path)
        if os.path.exists(full_path):
            return full_path
            
    # Try stripping quotes if present
    if path.startswith("'") and path.endswith("'"):
        return resolve_file_path(path[1:-1], base_paths)
    if path.startswith('"') and path.endswith('"'):
        return resolve_file_path(path[1:-1], base_paths)
            
    # Couldn't find the file
    return None

# Update the main function to enable combined mode
async def main():
    # Set up environment variables
    if not setup_environment():
        return
    
    # Configure the graph with a persistent thread_id for tracking in LangSmith
    thread_id = str(uuid.uuid4())
    print(f"Thread ID for LangSmith tracking: {thread_id}")
    
    thread = {
        "configurable": {
            "thread_id": thread_id,
            "search_api": "tavily",
            "planner_provider": "openai",
            "planner_model": "gpt-4o",
            "writer_provider": "openai",
            "writer_model": "gpt-4o",
            "max_search_depth": 1,
        }
    }
    
    # First, let's check for some common file references to help with debugging
    print("\nChecking for typical files in your project:")
    test_paths = [
        "source_docs/odr_digital_tvilling.txt",
        "source_docs/odr_vinnova.txt"
    ]
    for path in test_paths:
        resolved_path = resolve_file_path(path)
        if resolved_path:
            print(f"✅ Found file: {path}")
            print(f"   Full path: {os.path.abspath(resolved_path)}")
            with open(resolved_path, 'r', encoding='utf-8') as f:
                print(f"   Size: {len(f.read())} characters")
        else:
            print(f"❌ File not found: {path}")
            print(f"   Looked in: {', '.join(['current directory'] + ['source_docs', 'data', 'files', 'resources'])}")
    
    # Ask for research topic - now with COMBINED FILE SUPPORT ENABLED
    topic = get_user_input("What topic would you like to research?", 
                          allow_file=True, 
                          allow_combined=True,  # This is key! Enable combined mode
                          debug=True)
    
    if not topic:
        print("No topic entered. Exiting.")
        return
    
    print(f"\n📝 Research Topic: {topic if len(topic) < 100 else topic[:97] + '...'}")
    
    # Set up the graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # Variable to store the final report content
    final_report = None
    
    try:
        # Generate report plan
        print("\n--- STEP 1: GENERATING REPORT PLAN ---\n")
        plan_generated = False
        plan_shown = False
        
        # First run with the topic input
        async for event in graph.astream({"topic": topic}, thread, stream_mode="updates"):
            if event:
                # Try to format the event if it's the report plan
                if not plan_shown:
                    plan_shown = format_report_plan(event)
                
                if not plan_shown:
                    # If we couldn't format it as a plan, just print the raw event
                    print(event)
                    print("\n")
                
                plan_generated = True
        
        if not plan_generated:
            print("No report plan was generated. There might be an issue with the API.")
            return
            
        # Get feedback on the plan
        while True:
            feedback_choice = get_user_input("Would you like to provide feedback on the report plan? (yes/no)")
            if feedback_choice.lower() in ["n", "no"]:
                # User is satisfied with the plan, proceed to generation
                proceed = True
                break
            elif feedback_choice.lower() in ["y", "yes"]:
                # Get specific feedback with file support and multi-line input
                print("\nEnter your feedback (you can use 'file:path/to/file.txt' to read from a file):")
                feedback = get_user_input("For multi-line input directly, type your text and end with 'EOF' on a new line", 
                                         allow_file=True, multiline=True)
                
                print("\n--- UPDATING REPORT PLAN BASED ON FEEDBACK ---\n")
                feedback_applied = False
                plan_shown = False
                
                # Next run with the feedback
                async for event in graph.astream(Command(resume=feedback), thread, stream_mode="updates"):
                    if event:
                        # Try to format the event if it's the updated report plan
                        if not plan_shown:
                            plan_shown = format_report_plan(event)
                        
                        if not plan_shown:
                            # If we couldn't format it as a plan, just print the raw event
                            print(event)
                            print("\n")
                        
                        feedback_applied = True
                
                if not feedback_applied:
                    print("Failed to apply feedback. There might be an issue with the API.")
                    return
            else:
                print("Invalid choice. Please enter 'yes' or 'no'.")
                continue
                
        # Confirm to proceed with report generation
        while True:
            confirm = get_user_input("Generate the full report now? This may take several minutes. (yes/no)")
            
            if confirm.lower() in ["y", "yes"]:
                break
            elif confirm.lower() in ["n", "no"]:
                print("Report generation cancelled. Exiting.")
                return
            else:
                print("Invalid choice. Please enter 'yes' or 'no'.")
        
        # Generate the report
        print("\n--- GENERATING FULL REPORT ---\n")
        print("This may take several minutes depending on the complexity of the topic and depth of research.\n")
        
        # Final run to generate the report
        async for event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
            if event:
                # Try to format the event if it's report content
                formatted = format_report_content(event)
                
                if not formatted:
                    # If we couldn't format it as report content, just print the raw event
                    print(event)
                    print("\n")
                
                # Capture the final report if available
                if isinstance(event, dict) and 'final_report' in event:
                    final_report = event['final_report']
        
        print("\n✅ Report generation complete!")
        
        # Save report to file if we have content
        if final_report:
            save_path = save_report_to_file(final_report, topic, thread_id)
            if save_path:
                print(f"\n📄 Report saved to: {save_path}")
        
        print(f"\nThread ID for LangSmith reference: {thread_id}")
    
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {str(e)}")
        print(f"Message: {str(e)}")
        if "authentication_error" in str(e).lower() or "api key" in str(e).lower():
            print("\nAuthentication error detected. Please check that your API keys are valid.")
            print("Make sure your .env file contains valid keys for all required services.")
        # Print more detailed stack trace for debugging
        import traceback
        print("\nDetailed error information:")
        traceback.print_exc()

# Run the async function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting...")
        sys.exit(0) 