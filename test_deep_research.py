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
    
    if missing_keys:
        print("ERROR: The following API keys are missing or invalid in your .env file:")
        for key in missing_keys:
            print(f"  - {key}")
        print("\nPlease set valid API keys in your .env file and run the script again.")
        print("Required keys: " + ", ".join(required_keys))
        return False
    
    print("✅ All environment variables set successfully!")
    return True

def get_user_input(prompt):
    """Get input from the user with a prompt."""
    print(f"\n{prompt}")
    return input("> ").strip()

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
    
    # Ask for research topic
    topic = get_user_input("What topic would you like to research?")
    if not topic:
        print("No topic entered. Exiting.")
        return
    
    print(f"\n📝 Research Topic: {topic}")
    
    # Set up the graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
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
                # Get specific feedback
                feedback = get_user_input("Please provide your feedback on the report plan:")
                
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
                
        print("\n✅ Report generation complete!")
        print(f"Thread ID for LangSmith reference: {thread_id}")
    
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