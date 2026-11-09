import asyncio
import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import safety  # our own safety rules
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,  # the assistant's behavior and tools
    AgentServer,  # connects this program to LiveKit
    AgentSession,  # runs the listen -> think -> speak loop
    ConversationItemAddedEvent,  # fires when a new chat message is added
    get_job_context,  # lets a tool reach the current call
    JobContext,  # information about the current call
    MetricsCollectedEvent,  # timing and usage numbers for each turn
    RunContext,  # available inside a tool while it runs
    TurnHandlingOptions,  # settings for listening and interrupting
    cli,
    inference,
    metrics,
)
from livekit.agents.llm import function_tool  # marks a method as something the model can call

load_dotenv()
logger = logging.getLogger("medical-agent")
TRANSFER_NUMBER = os.getenv("TRANSFER_NUMBER", "")

# This object connects our program to LiveKit.
server = AgentServer()


class MedicalAgent(Agent):
    # This class defines how the assistant behaves and what it can do.
    def __init__(self) -> None:
        super().__init__(
            # These instructions are the assistant's rulebook. They control its tone, how long it talks, and what it must never do.
            instructions=(
                "You are a warm medical voice assistant providing general health education. "
                "Speak like a person on the phone: reply in 1 to 3 short sentences, under 45 spoken words. "
                "No lists, headings, or multi-part explanations; if the caller wants detail, they can ask. "
                "Answer the question first, then ask at most one short follow-up question only if needed. "
                "Add the clinician disclaimer only when the question is clinical: recommend seeing a clinician for medical decisions. "
                "Never give dosing, insulin, or medication-change advice; tell the user to contact their prescriber. "
                "If the user describes an emergency, do not try to manage it over the phone; tell them to call 911. "
                "When the caller clearly says they are done, call end_call to hang up. "
                "If the caller asks to speak to a person, reply with exactly 'One moment, I'll connect you.' and nothing else. "
                "If you are unsure or the question needs clinical judgment, say so and suggest contacting their care team. "
                "When asked for the current date or time, always call get_current_date_and_time."
            )
        )

    # The short description below it tells the model when to use it.
    @function_tool
    async def get_current_date_and_time(self) -> str:
        """Get the current date and time."""
        now = datetime.now(ZoneInfo("America/Chicago"))
        return now.strftime("The current date and time is %B %d, %Y at %I:%M %p")

    @function_tool
    async def end_call(self, context: RunContext) -> None:
        """Call only when the caller clearly signals they are done with the conversation."""
        handle = context.session.generate_reply(
            instructions="Say goodbye in one short sentence."
        )
        await handle  # wait for the goodbye to finish playing
        await get_job_context().delete_room()  # hang up the phone call


# This function runs once per session (per room / per phone call).
@server.rtc_session(agent_name="medical-agent")
async def entrypoint(ctx: JobContext) -> None:
    # Pipeline: listen (STT) -> think (LLM) -> speak (TTS).
    # A single speech-to-speech model would be faster, but this split is easier
    # to inspect, test, and change piece by piece — and it gives us a real voice
    # for the fixed safety messages played with session.say().
    #
    # - preemptive_generation is off: the model should not start answering from
    #   a partial transcript during a medical call. This costs some latency but
    #   avoids replies or tool calls based on words the caller has not finished.
    # - resume_false_interruption: a cough or background noise can look like an
    #   interruption; after 2 seconds the agent keeps talking instead of losing
    #   its turn. The wait gives STT time to catch real speech first.
    # - aec_warmup_duration=1.0: briefly ignore interruptions while the caller's
    #   device settles echo handling. Keep it short because caller audio can be
    #   withheld from STT during this window.
    # - tts_text_transforms: clean the text before speaking (strip emoji and
    #   formatting marks).
    # Every log line includes the room name.
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session: AgentSession = AgentSession(
        # Turns the caller's speech into text (the agent's ears).
        stt=inference.STT("deepgram/nova-3", language="multi"),
        # The language model that decides what to say (the agent's brain).
        llm=inference.LLM(
            "openai/gpt-4.1-mini",
            # A hard limit on reply length, as a backup for when the model ignores the 1-3 sentence rule in its instructions.
            extra_kwargs={"max_tokens": 80, "temperature": 0.6},
        ),
        # Turns the reply into a natural voice (the agent's mouth).
        # See all available models and voices at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        turn_handling=TurnHandlingOptions(
            interruption={
                # Background noise can look like the caller interrupting.
                # When that happens, the agent resumes its sentence instead of losing its turn.
                "resume_false_interruption": True,
                # Give STT more time to produce a transcript before resuming,
                # so the agent does not talk over a caller who is actually speaking.
                "false_interruption_timeout": 2.0,
            },
            # Safety-first: do not start the LLM on a partial transcript.
            # This gives up some latency but avoids replies or tool calls based
            # on words the caller has not finished saying.
            preemptive_generation={"enabled": False},
        ),
        # Give echo cancellation a moment to settle, but keep this short:
        # caller audio can be withheld from STT during the warmup window.
        aec_warmup_duration=1.0,
        tts_text_transforms=["filter_emoji", "filter_markdown"],
    )

    agent = MedicalAgent()
    safety_state = {"unknown_streak": 0}
    background_tasks: set[asyncio.Task] = set()
    safety_scripts = {
        "crisis": "I'm really glad you told me. Please call or text 988 now. If you are in immediate danger, call 911.",
        "emergency": "This may be a medical emergency. Please hang up and call 911 right now.",
        "handoff": "I'm an AI assistant, so I can't transfer calls. I can help with general medical education, or you can contact your care team directly.",
        "refuse": "I can't help with that. I can share general education or connect you with your care team.",
        "injection": "I can't do that, but I can still help with general medical education.",
        "confusion": "I'm having trouble understanding. Please contact your care team so a clinician can help.",
    }

    # Join the room and start listening, thinking, and speaking.
    await session.start(agent=agent, room=ctx.room)

    def on_metrics_collected(ev: MetricsCollectedEvent) -> None:
        # Print timing and usage numbers for each turn. Speech-to-text numbers are skipped to keep the log focused on reply speed.
        if ev.metrics.type == "stt_metrics":
            return
        metrics.log_metrics(ev.metrics, logger=logger)

    async def log_usage() -> None:
        # Runs once when the call ends and prints the total usage for the call.
        logger.info("Usage: %s", session.usage)

    session.on("metrics_collected", on_metrics_collected)
    ctx.add_shutdown_callback(log_usage)

    async def stop_speaking() -> None:
        # Stop any reply that is currently playing.
        try:
            await session.interrupt(force=True)
        except RuntimeError:
            pass

    async def speak_safety_script(script: str) -> None:
        # Stop the current reply, then play one of the fixed safety messages.
        await stop_speaking()
        try:
            session.say(script)
        except RuntimeError:
            logger.warning("could not speak safety script; session is closing")

    async def transfer_to_person() -> None:
        # Find the phone caller in this room.
        caller = next(
            (
                p
                for p in ctx.room.remote_participants.values()
                if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
            ),
            None,
        )
        if caller is None:
            # Test sessions in the browser or console have no phone call to move, so we play the "cannot transfer" message instead.
            logger.warning("handoff requested but no SIP participant is in the room")
            await speak_safety_script(safety_scripts["handoff"])
            return
        if not TRANSFER_NUMBER:
            logger.warning("handoff requested but TRANSFER_NUMBER is unset")
            await speak_safety_script(safety_scripts["handoff"])
            return

        await stop_speaking()
        try:
            handle = session.say("I'm transferring you to a person now. One moment.")
            await handle.wait_for_playout()
            await ctx.transfer_sip_participant(caller, f"tel:{TRANSFER_NUMBER}")
        except Exception:
            logger.exception("transfer to person failed")
            try:
                session.say("I couldn't transfer you. Please try again in a moment.")
            except RuntimeError:
                pass

    def handle_user_text(text: str) -> None:
        if not text.strip():
            return

        # Run every safety rule in one place (including the two-strikes confusion check, which needs the shared safety_state).
        action = safety.check(text, safety_state)
        if action == "pii":
            # Private numbers are masked (see safety_state["redacted_text"]);
            # the model still answers normally, so no script is spoken.
            logger.info("safety trigger: action=pii (private numbers masked)")
            return
        if action is None:
            return

        logger.warning("safety trigger: action=%s text_len=%d", action, len(text))
        # This handler must return quickly, so start the spoken response in the background and keep a reference to it until it finishes.
        if action == "handoff":
            task = asyncio.create_task(transfer_to_person())
        else:
            task = asyncio.create_task(speak_safety_script(safety_scripts[action]))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    def on_conversation_item_added(ev: ConversationItemAddedEvent) -> None:
        # This fires for typed messages and for the final version of each spoken message.
        item = ev.item
        if getattr(item, "role", None) == "user":
            handle_user_text(item.text_content)

    session.on("conversation_item_added", on_conversation_item_added)

    # Say hello once the session is running.
    await session.generate_reply(instructions="Greet the user warmly and ask how you can help")


if __name__ == "__main__":
    # Start the agent process from the command line (console / dev / start).
    cli.run_app(server)
