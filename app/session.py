import os
import random
import string
import json
from dotenv import load_dotenv

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import Session, InMemorySessionService

from .agent_team.agent import root_agent

load_dotenv("./env")
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') else 'No'}")
print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
print(f"Anthropic API Key set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")

UNIFIED_CONVERSATION_ID = "one_conversation"

class Conversation:
    def __init__(self, user_id: str, runner: Runner):
        self.user_id = user_id
        self.runner = runner

class ConversationOverloadError(RuntimeError):
    pass

class InvalidConversationError(ValueError):
    pass

# Thanks to
# https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits
def random_string_id(length: int):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class ConversationManager:
    def __init__(self, app_name: str, max_sessions: int):
        self.app_name = app_name
        self.MAX_CONVERSATION_COUNT = max_sessions
        self.conversations: dict[Conversation] = {}
        self.session_service = InMemorySessionService()

    async def init_conversation(self, user_id: str) -> None:
        if len(self.conversations) > self.MAX_CONVERSATION_COUNT:
            raise ConversationOverloadError()
        if (not (user_id in self.conversations)):
            runner = await self.create_runner(user_id, agent=root_agent)
            convo = Conversation(
                user_id,
                runner
            )
            self.conversations[user_id] = convo
    
    async def push_conversation(self, user_id: str, query: str):
        if (not (user_id in self.conversations)):
            raise InvalidConversationError() 
        convo = self.conversations[user_id]
        assert convo.user_id == user_id
        
        return await self.call_agent_async(convo.runner, user_id, query)
    
    async def generate_conversation(self, user_id: str, query: str):
        if (not (user_id in self.conversations)):
            raise InvalidConversationError() 
        convo = self.conversations[user_id]
        assert convo.user_id == user_id
        
        return self.generate_agent_async(convo.runner, user_id, query)

    def close_conversation(self, user_id: str):
        if (not (user_id in self.conversations)):
            raise InvalidConversationError() 
        convo = self.conversations.pop(user_id)
        
        self.session_service.delete_session(
            app_name=self.app_name, # Use the consistent app name
            user_id=user_id,
            session_id=UNIFIED_CONVERSATION_ID
        )
        
        del convo

    """
    Session code adapted from Google ADK tutorial
    """
    async def create_runner(self, user_id: str, agent) -> Runner:
        """
        Creates a new conversation with a multi-agent AI, using in memory session.

        Args:
            conversation_id (str): An identifier for the session
            user_id         (str): An identifier for the user
        Returns:
            Runner: The Runner used to interact with the conversation
        """
        # Define initial state data
        initial_state = {
            "user:mood": "Neutral",
            "user_preference_temperature_unit": "Celsius"
        }

        # Create the session, providing the initial state
        session_stateful = self.session_service.create_session(
            app_name=self.app_name, # Use the consistent app name
            user_id=user_id,
            session_id=UNIFIED_CONVERSATION_ID,
            state=initial_state # <<< Initialize state during creation
        )
        print(f"✅ Session '{UNIFIED_CONVERSATION_ID}' created for user '{user_id}'.")

        # Verify the initial state was set correctly
        retrieved_session = self.session_service.get_session(app_name=self.app_name,
                                                                user_id=user_id,
                                                                session_id = UNIFIED_CONVERSATION_ID)
        if (not retrieved_session):
            raise RuntimeError("Could not create new session!")

        # --- Create Runner for this Agent, Using SAME Stateful Session Service ---
        runner_root_tool_guardrail = Runner(
            agent=agent,
            app_name=self.app_name, # Use consistent APP_NAME
            session_service=self.session_service
        )
        print(f"✅ Runner created for tool guardrail agent '{runner_root_tool_guardrail.agent.name}', using stateful session service.")
        return runner_root_tool_guardrail

    """
    Runner code adapted from Google ADK tutorial
    """
    async def call_agent_async(self, runner, user_id, query: str) -> str:
        """Sends a query to the agent and prints the final response."""
        print(f"\n>>> User Query: {query}")

        # Prepare the user's message in ADK format
        content = types.Content(role='user', parts=[types.Part(text=query)])

        final_response_text = "Agent did not produce a final response." # Default

        # Key Concept: run_async executes the agent logic and yields Events.
        # We iterate through events to find the final answer.
        async for event in runner.run_async(user_id=user_id, session_id=UNIFIED_CONVERSATION_ID, new_message=content):
            # You can uncomment the line below to see *all* events during execution
            # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

            # Key Concept: is_final_response() marks the concluding message for the turn.
            if event.is_final_response():
                if event.content and event.content.parts:
                    # Assuming text response in the first part
                    final_response_text = event.content.parts[0].text
                elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                    final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                # Add more checks here if needed (e.g., specific error codes)
                return final_response_text # Stop processing events once the final response is found
    
    async def generate_agent_async(self, runner: Runner, user_id, query: str):
        """Sends a query to the agent and streams any resulting events."""
        print(f"\n>>> User Query: {query}")

        # Prepare the user's message in ADK format
        content = types.Content(role='user', parts=[types.Part(text=query)])

        session: Session = await self.session_service.get_session(
            app_name=self.app_name,
            user_id=user_id, 
            session_id=UNIFIED_CONVERSATION_ID
        )

        # Key Concept: run_async executes the agent logic and yields Events.
        async for event in runner.run_async(user_id=user_id, session_id=UNIFIED_CONVERSATION_ID, new_message=content):
            print(f"[{event.timestamp}] Event. Author: {event.author}")
            if event.content and event.content.parts:
                if not event.partial:
                    mood = session.state['user:mood']
                    if event.actions and event.actions.state_delta:
                        if 'user:mood' in event.actions.state_delta:
                            mood = event.actions.state_delta['user:mood']
                    print(f"Current mood is : {mood}")
                    try:
                        msg = {
                            "parts": event.content.to_json_dict()["parts"],
                            "author": event.author,
                            "mood": mood,
                            "is_final": event.is_final_response()
                        }
                        yield json.dumps(msg) + "\n\n"
                    except Exception as e:
                        print(f"Failed to dump event to json!, {e}")
            elif event.actions and event.actions.escalate: # Handle potential errors/escalations
                yield f"Agent escalated: {event.error_message or 'No specific message.'}\n\n"
