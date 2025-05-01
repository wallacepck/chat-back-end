from typing import Self
import os
import random
import string
from dotenv import load_dotenv

from google.genai import types
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from google_adk_tutorial.multi_tool_agent.agent import root_agent

load_dotenv("./env")
print("API Keys Set:")
print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') else 'No'}")
print(f"OpenAI API Key set: {'Yes' if os.environ.get('OPENAI_API_KEY') else 'No'}")
print(f"Anthropic API Key set: {'Yes' if os.environ.get('ANTHROPIC_API_KEY') else 'No'}")

class Conversation:
    def __init__(self, conversation_id: str, user_id: str, runner: Runner):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.runner = runner

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

    def new_conversation(self, user_id: str) -> str:
        if len(self.conversations) > self.MAX_CONVERSATION_COUNT:
            raise RuntimeError("Cannot create new conversation, too many ongoing conversations!")

        conversation_id = random_string_id(16)
        # The collision probability is very low
        while (conversation_id in self.conversations):
            conversation_id = random_string_id(16)

        runner = self.create_runner(conversation_id, user_id, agent=root_agent)

        convo = Conversation(
            conversation_id,
            user_id,
            runner
        )
        self.conversations[conversation_id] = convo

        return convo.conversation_id
    
    async def push_conversation(self,  conversation_id: str, user_id: str, query: str):
        if (not (conversation_id in self.conversations)):
            raise RuntimeError("No such conversation!") 
        convo = self.conversations[conversation_id]
        if (convo.user_id != user_id):
            raise RuntimeError("No such conversation!")
        
        return await self.call_agent_async(convo.runner, user_id, convo.conversation_id, query)

    def close_conversation(self,  conversation_id: str, user_id: str):
        if (not (conversation_id in self.conversations)):
            raise RuntimeError("No such conversation!") 
        convo = self.conversations[conversation_id]
        if (convo.user_id != user_id):
            raise RuntimeError("No such conversation!")
        
        self.session_service.delete_session(
            app_name=self.app_name, # Use the consistent app name
            user_id=user_id,
            session_id=conversation_id
        )
        
        del convo

    """
    Session code adapted from Google ADK tutorial
    """
    def create_runner(self, conversation_id: str, user_id: str, agent) -> Runner:
        """
        Creates a new conversation with a multi-agent AI, using in memory session.

        Args:
            conversation_id (str): An identifier for the session
            user_id         (str): An identifier for the user
        Returns:
            Runner: The Runner used to interact with the conversation
        """
        # Define initial state data - user prefers Celsius initially
        initial_state = {
            "user_preference_temperature_unit": "Celsius"
        }

        # Create the session, providing the initial state
        session_stateful = self.session_service.create_session(
            app_name=self.app_name, # Use the consistent app name
            user_id=user_id,
            session_id=conversation_id,
            state=initial_state # <<< Initialize state during creation
        )
        print(f"✅ Session '{conversation_id}' created for user '{user_id}'.")

        # Verify the initial state was set correctly
        retrieved_session = self.session_service.get_session(app_name=self.app_name,
                                                                user_id=user_id,
                                                                session_id = conversation_id)
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
    async def call_agent_async(self, runner, user_id, session_id, query: str) -> str:
        """Sends a query to the agent and prints the final response."""
        print(f"\n>>> User Query: {query}")

        # Prepare the user's message in ADK format
        content = types.Content(role='user', parts=[types.Part(text=query)])

        final_response_text = "Agent did not produce a final response." # Default

        # Key Concept: run_async executes the agent logic and yields Events.
        # We iterate through events to find the final answer.
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
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