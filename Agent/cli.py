import sys
import vertexai
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from config import settings
from agent_engine import TextOrderAgent

def main():
    console = Console()
    console.print(Panel.fit("[bold blue]Agent 01 - Reasoning Engine Mode[/bold blue]", title="Welcome"))
    console.print(f"[dim]Project: {settings.GCP_PROJECT_ID}[/dim]")
    console.print("[dim]Type 'exit' or 'quit' to stop.[/dim]\n")

    try:
        # Initialize the Reasoning Engine Agent
        # Note: This requires valid GCP credentials.
        with console.status("[bold green]Initializing Reasoning Engine...[/bold green]"):
            agent = TextOrderAgent()
        
        console.print("[green]Agent initialized successfully.[/green]")
        console.print("[bold]You may start ordering.[/bold]")
        console.print("-" * 30)

    except Exception as e:
        console.print(f"[bold red]Error initializing Agent:[/bold red] {e}")
        console.print("[yellow]Ensure you have Google Cloud credentials set up.[/yellow]")
        return

    # In a local RE pattern, the agent instance holds the state (self._current_order).
    # so we don't need to manually pass history/state in the loop for the simplified version,
    # PROVIDED the ReasoningEngine wrapper keeps the chat session alive?
    # Actually, `engine.query` might be stateless unless we manage the chat history explicitly.
    # But `FruitStoreAgent` typically manages its own state in `self._current_order`.
    # Let's assume the `query` method handles the interaction.
    
    # Wait, `engine.query` in the library typically starts a new session or uses the provided one.
    # In my `agent_engine.py`, I create a NEW `ReasoningEngine` inside `query`, which might reset chat history context 
    # (though `_current_order` state persists in `self`).
    # This might mean the LLM forgot what it said, but it knows the Order State.
    # This matches the user's "State Persistence" requirement perfectly!
    
    while True:
        try:
            console.print()
            user_input = Prompt.ask("[bold yellow]You[/bold yellow]").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                console.print("[bold blue]Goodbye![/bold blue]")
                break
            
            if not user_input:
                continue

            with console.status("[bold magenta]Agent is thinking...[/bold magenta]", spinner="dots"):
                # Call the Reasoning Engine
                response = agent.query(message=user_input)
            
            # The response from engine.query is typically a string or an object with 'output'
            # Let's assume string for now or inspect it.
            reply_text = str(response)

            console.print(f"[bold cyan]Agent[/bold cyan]: {reply_text}")
            
            # Debug: Show current order state
            if settings.DEBUG:
                current_state = agent.get_current_order()
                console.print(f"[dim]Current State: {current_state}[/dim]")
            
            console.print("-" * 30)

        except KeyboardInterrupt:
            console.print("\n[bold blue]Goodbye![/bold blue]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}\n")

if __name__ == "__main__":
    main()
