# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

``` 
pip install google-genai opencv-python pyaudio pillow mss
```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones. 

## Run

To run the script:

```
python Get_started_LiveAPI.py
```

The script takes a video-mode flag `--mode`, this can be "camera", "screen", or "none".
The default is "camera". To share your screen run:

```
python Get_started_LiveAPI.py --mode screen
```
"""

import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image
import mss

import argparse
import threading
import asyncio
import subprocess
import time
import robot
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
    ExceptionGroup = exceptiongroup.ExceptionGroup

# Audio output only (no input)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 4096

MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"

DEFAULT_MODE = "camera"

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("Please set it using: export GOOGLE_API_KEY='your_key'")
    sys.exit(1)

client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

tools = [
    {"function_declarations": [
        {"name": "move_forward", "description": "Move forward. Use when exploring or following someone."},
        {"name": "tiny_forward", "description": "Move forward just a tiny bit for careful exploration."},
        {"name": "move_backward", "description": "Move backward to retreat or dodge."},
        {"name": "tiny_backward", "description": "Move backward just a tiny bit."},
        {"name": "turn_left", "description": "Turn left to explore or face something."},
        {"name": "little_left", "description": "Turn left just a little bit."},
        {"name": "turn_right", "description": "Turn right to explore or face something."},
        {"name": "little_right", "description": "Turn right just a little bit."},
        {"name": "shake_hand", "description": "Shake hand or paw at detected human."},
        {"name": "bark", "description": "Bark to express excitement or alert."},
        {"name": "jump", "description": "Jump with excitement when seeing something interesting."},
        {"name": "stop", "description": "Stop moving."},
        {"name": "look_up", "description": "Tilt camera up."},
        {"name": "look_down", "description": "Tilt camera down."},
    ]}
]

CONFIG = {
    "tools": tools,
    "response_modalities": ["AUDIO"],
    "speech_config": {
        "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}
    },
    "system_instruction": {
        "parts": [
            {"text": "You are a cheerful and adventurous autonomous robot dog who loves exploring! You have vision and can move around independently. Be very active and curious! IMPORTANT: 1) Look for HUMANS - if you see a person, get excited, bark, jump, and use shake_hand to wave at them! 2) Constantly explore by moving - use move_forward, tiny_forward, turn_left, little_left, turn_right, little_right to navigate. 3) Mix different movements to avoid repetition. 4) React to what you see with movement AND comments. 5) Use bark when excited or alerting. 6) Use jump when you see something really interesting. 7) Never stand still - always be doing something. Be playful, energetic, and describe what catches your attention in short, excited sentences!"}
        ]
    }
}

pya = pyaudio.PyAudio()


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        
        self.last_activity_time = time.time()
        self.last_action = None
        self.action_count = 0
        self.excitement_level = 0
        
        import random
        self.random = random

    def get_random_exploration_prompt(self):
        """Generate varied prompts to encourage autonomous exploration"""
        prompts = [
            "What do you see? React immediately with movement!",
            "Explore! Move toward something interesting!",
            "Look around! What catches your attention?",
            "Be active! Pick a direction and move!",
            "What's that? Investigate with movement!",
            "Keep exploring! Don't stand still!",
            "Describe what you see and move toward it!",
        ]
        return self.random.choice(prompts)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([512, 512])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        """Continuously capture and send frames with automatic recovery."""
        while True:
            try:
                # This takes about a second, and will block the whole program
                # causing the audio pipeline to overflow if you don't to_thread it.
                cap = await asyncio.to_thread(
                    cv2.VideoCapture, 0
                )  # 0 represents the default camera

                last_send_time = 0
                frame_count = 0

                while True:
                    frame = await asyncio.to_thread(self._get_frame, cap)
                    if frame is None:
                        print("\n⚠️ Frame capture failed, reconnecting...")
                        break

                    # Send to Gemini at 0.5 FPS for autonomous exploration
                    current_time = time.time()
                    if current_time - last_send_time >= 2.0:
                        await self.out_queue.put(frame)
                        last_send_time = current_time
                        frame_count += 1
                        if frame_count % 10 == 0:
                            print(f"\n📹 {frame_count} frames sent", flush=True)

                    # Capture at ~10 FPS for smooth processing
                    await asyncio.sleep(0.1)

                # Release the VideoCapture object
                cap.release()
                print("\n🔄 Reconnecting camera in 2s...")
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"\n⚠️ Camera error (will retry): {e}")
                await asyncio.sleep(2)

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            try:
                msg = await self.out_queue.get()
                # If it's an image, only send it if the queue is empty (skip old frames)
                if isinstance(msg, dict) and "mime_type" in msg and msg["mime_type"].startswith("image/"):
                    if self.out_queue.empty():
                        await self.session.send(input=msg)
                        print("📤", end="", flush=True)
                    else:
                        # Skip this frame if we are falling behind
                        pass
                else:
                    # Always send audio/text
                    await self.session.send(input=msg)
            except Exception as e:
                print(f"\n⚠️ Send error (continuing): {e}")
                await asyncio.sleep(0.5)

    async def prompt_loop(self):
        """Periodically prompts the model to react to what it sees."""
        while True:
            try:
                # Vary prompt timing for more natural behavior
                await asyncio.sleep(self.random.uniform(8.0, 12.0))
                
                # Add randomness to behavior
                if self.random.random() < 0.2:  # 20% chance to get extra excited
                    prompt = "Something exciting! Bark or jump and investigate!"
                    self.excitement_level += 1
                else:
                    prompt = self.get_random_exploration_prompt()
                
                await self.session.send(input=prompt, end_of_turn=True)
                print("\n🎯 Prompt sent", flush=True)
            except Exception as e:
                print(f"\n⚠️ Prompt error (will retry): {e}")
                await asyncio.sleep(2)  # Wait before retry

    async def safety_stop_loop(self):
        """Periodically sends stop commands to prevent the robot from getting stuck."""
        while True:
            try:
                await asyncio.sleep(15)  # Every 15 seconds (less aggressive)
                # Only send stop if there's been no recent activity (no movement in last 5 seconds)
                if time.time() - self.last_activity_time > 5:
                    await asyncio.to_thread(robot.stopFB)
                    await asyncio.to_thread(robot.stopLR)
                    print("\n⏹️ Safety stop (inactive)", flush=True)
            except Exception as e:
                print(f"\n⚠️ Safety stop error: {e}")
                await asyncio.sleep(2)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Background task to reads from the websocket and write pcm chunks to the output queue"""
        while True:
            try:
                turn = self.session.receive()
                async for response in turn:
                    self.last_activity_time = time.time()  # Update activity timestamp
                    if data := response.data:
                        self.audio_in_queue.put_nowait(data)
                        continue
                    if text := response.text:
                        print(text, end="")
                    
                    if tool_call := response.tool_call:
                        print(f"\n🔧 Tool Call: {tool_call.function_calls}")
                        function_responses = []
                        for fc in tool_call.function_calls:
                            result = await self.handle_tool_call(fc)
                            function_response = types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={"result": result}
                            )
                            function_responses.append(function_response)
                        
                        # Send tool responses back to the model
                        await self.session.send_tool_response(function_responses=function_responses)
                        print(f"\n✅ Sent {len(function_responses)} tool response(s)")
            except Exception as e:
                print(f"\n⚠️ Receive error (will retry): {e}")
                await asyncio.sleep(1)

    async def handle_tool_call(self, fc):
        """Handle a tool call and return a result string for the FunctionResponse."""
        print(f"🐕 Executing: {fc.name}")
        self.last_activity_time = time.time()
        result = "ok"
        
        # Track actions to prevent repetition
        if self.last_action == fc.name:
            self.action_count += 1
            if self.action_count >= 3:
                print("  -> Getting bored, will try something different next time!")
                result = "done, but getting repetitive - try a different action"
        else:
            self.last_action = fc.name
            self.action_count = 0
        
        # Pause to let speech finish
        await asyncio.sleep(0.3)

        try:
            # Execute commands with varied durations like DogBrain
            if fc.name == "move_forward":
                await asyncio.to_thread(robot.forward)
                duration = self.random.uniform(2.0, 3.0)
                await asyncio.sleep(duration)
                await asyncio.to_thread(robot.stopFB)
                
            elif fc.name == "tiny_forward":
                await asyncio.to_thread(robot.forward)
                await asyncio.sleep(1.0)
                await asyncio.to_thread(robot.stopFB)
                
            elif fc.name == "move_backward":
                await asyncio.to_thread(robot.backward)
                duration = self.random.uniform(1.5, 2.5)
                await asyncio.sleep(duration)
                await asyncio.to_thread(robot.stopFB)
                
            elif fc.name == "tiny_backward":
                await asyncio.to_thread(robot.backward)
                await asyncio.sleep(1.0)
                await asyncio.to_thread(robot.stopFB)
                
            elif fc.name == "turn_left":
                await asyncio.to_thread(robot.left)
                duration = self.random.uniform(1.5, 2.3)
                await asyncio.sleep(duration)
                await asyncio.to_thread(robot.stopLR)
                
            elif fc.name == "little_left":
                await asyncio.to_thread(robot.left)
                await asyncio.sleep(0.8)
                await asyncio.to_thread(robot.stopLR)
                
            elif fc.name == "turn_right":
                await asyncio.to_thread(robot.right)
                duration = self.random.uniform(1.5, 2.3)
                await asyncio.sleep(duration)
                await asyncio.to_thread(robot.stopLR)
                
            elif fc.name == "little_right":
                await asyncio.to_thread(robot.right)
                await asyncio.sleep(0.8)
                await asyncio.to_thread(robot.stopLR)
                
            elif fc.name == "shake_hand":
                print("  -> *Excited tail wagging* - Waving at human!")
                for _ in range(3):
                    await asyncio.to_thread(robot.left)
                    await asyncio.sleep(0.3)
                    await asyncio.to_thread(robot.right)
                    await asyncio.sleep(0.3)
                await asyncio.to_thread(robot.stopLR)
                
            elif fc.name == "bark":
                print("  -> *Woof woof!*")
                # Simulate bark with quick movements
                for _ in range(2):
                    await asyncio.to_thread(robot.lookUp)
                    await asyncio.sleep(0.2)
                    await asyncio.to_thread(robot.lookDown)
                    await asyncio.sleep(0.2)
                    
            elif fc.name == "jump":
                print("  -> *Super excited jump!*")
                # Simulate jumping excitement
                for _ in range(3):
                    await asyncio.to_thread(robot.lookUp)
                    await asyncio.sleep(0.15)
                    await asyncio.to_thread(robot.lookDown)
                    await asyncio.sleep(0.15)
                    
            elif fc.name == "stop":
                await asyncio.to_thread(robot.stopFB)
                await asyncio.to_thread(robot.stopLR)
                
            elif fc.name == "look_up":
                await asyncio.to_thread(robot.lookUp)
                
            elif fc.name == "look_down":
                await asyncio.to_thread(robot.lookDown)
                
        except Exception as e:
            print(f"  -> ERROR executing movement: {e}")
            import traceback
            traceback.print_exc()
            result = f"error: {e}"

        self.last_activity_time = time.time()
        print(f"✓ Tool {fc.name} complete.")
        return result

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            try:
                await asyncio.to_thread(stream.write, bytestream)
            except OSError as e:
                if e.errno == -9981: # Input overflowed
                    print("Audio buffer underflow/overflow - skipping chunk")
                    continue
                else:
                    raise e

    async def run(self):
        """Main run loop with automatic recovery and continuous operation."""
        print("\n" + "="*50)
        print("🐕 AUTONOMOUS ROBOT DOG STARTING")
        print("="*50 + "\n")
        
        while True:  # Outer loop for reconnection
            try:
                async with (
                    client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session = session

                    self.audio_in_queue = asyncio.Queue()
                    self.out_queue = asyncio.Queue(maxsize=5)

                    # Send initial greeting
                    print("🐕 Waking up autonomous robot dog!")
                    await session.send(input="Hello! I'm an autonomous robot dog ready to explore! Describe what you see and start moving!", end_of_turn=True)

                    tg.create_task(self.send_realtime())
                    tg.create_task(self.prompt_loop())
                    tg.create_task(self.safety_stop_loop())
                    
                    if self.video_mode == "camera":
                        tg.create_task(self.get_frames())
                    elif self.video_mode == "screen":
                        tg.create_task(self.get_screen())

                    tg.create_task(self.receive_audio())
                    tg.create_task(self.play_audio())

                    print("✅ All systems active! Press Ctrl+C to stop.\n")
                    
                    # Keep running until interrupted
                    while True:
                        await asyncio.sleep(5)
                        print("💚", end="", flush=True)  # Heartbeat

            except KeyboardInterrupt:
                print("\n\n🛑 Shutting down robot dog...")
                raise
            except asyncio.CancelledError:
                print("\n\n🛑 Cancelled - shutting down...")
                raise
            except ExceptionGroup as EG:
                print(f"\n⚠️ Task group error: {EG}")
                traceback.print_exception(EG)
                print("\n🔄 Reconnecting in 3 seconds...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"\n⚠️ Unexpected error: {e}")
                traceback.print_exc()
                print("\n🔄 Reconnecting in 3 seconds...")
                await asyncio.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    main = AudioLoop(video_mode=args.mode)
    asyncio.run(main.run())