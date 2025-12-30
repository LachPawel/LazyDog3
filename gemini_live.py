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

load_dotenv()

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
    ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
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
        {"name": "move_forward", "description": "Move forward. Use this when the path is clear."},
        {"name": "move_backward", "description": "Move backward. Use this to retreat."},
        {"name": "turn_left", "description": "Turn left."},
        {"name": "turn_right", "description": "Turn right."},
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
            {"text": "You are an autonomous exploration robot. EVERY response MUST include: 1) One short sentence about what you see. 2) A movement function call. You have these functions: move_forward, turn_left, turn_right, move_backward. YOU MUST CALL A FUNCTION IN EVERY RESPONSE. No exceptions. Explore constantly."}
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

    def process_commands(self, text):
        pass

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

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
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

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
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def prompt_loop(self):
        """Periodically prompts the model to keep it active and moving."""
        while True:
            await asyncio.sleep(1)  # Check every second
            
            # If no activity for 10 seconds, prompt the model
            if time.time() - self.last_activity_time > 10:
                try:
                    print("\n[Auto-Prompting Model...]")
                    await self.session.send(input="View status and call a movement function now.", end_of_turn=True)
                    self.last_activity_time = time.time() # Reset timer to avoid double sending
                except Exception as e:
                    print(f"Keep-alive error: {e}")

    async def safety_stop_loop(self):
        """Periodically sends stop commands to prevent the robot from getting stuck."""
        while True:
            await asyncio.sleep(8)  # Every 8 seconds
            # Only send stop if there's been no recent activity (no movement in last 3 seconds)
            if time.time() - self.last_activity_time > 3:
                try:
                    await asyncio.to_thread(robot.stopFB)
                    await asyncio.to_thread(robot.stopLR)
                except Exception as e:
                    print(f"Safety stop error: {e}")

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
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                self.last_activity_time = time.time()  # Update activity timestamp
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                
                if tool_call := response.tool_call:
                    print(f"\nTool Call: {tool_call.function_calls}")
                    for fc in tool_call.function_calls:
                        await self.handle_tool_call(fc)

    async def handle_tool_call(self, fc):
        print(f"Executing tool: {fc.name}")
        self.last_activity_time = time.time()
        
        # Pause to let speech finish
        await asyncio.sleep(0.5)

        try:
            # Execute commands directly like in test_movement.py
            if fc.name == "move_forward":
                print("  -> Calling robot.forward()")
                await asyncio.to_thread(robot.forward)
                await asyncio.sleep(2.0)
                print("  -> Calling robot.stopFB()")
                await asyncio.to_thread(robot.stopFB)
            elif fc.name == "move_backward":
                print("  -> Calling robot.backward()")
                await asyncio.to_thread(robot.backward)
                await asyncio.sleep(2.0)
                await asyncio.to_thread(robot.stopFB)
            elif fc.name == "turn_left":
                print("  -> Calling robot.left()")
                await asyncio.to_thread(robot.left)
                await asyncio.sleep(1.0)
                await asyncio.to_thread(robot.stopLR)
            elif fc.name == "turn_right":
                print("  -> Calling robot.right()")
                await asyncio.to_thread(robot.right)
                await asyncio.sleep(1.0)
                await asyncio.to_thread(robot.stopLR)
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

        self.last_activity_time = time.time()
        print(f"Tool {fc.name} complete.")
        
        # Send a prompt to continue the conversation with speech after tool execution
        try:
            await self.session.send(input="Movement complete. Comment on what you see and choose next action.", end_of_turn=True)
        except Exception as e:
            print(f"Error sending continuation: {e}")

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
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=5)

                # Send initial greeting
                await session.send(input="Patrol mode active. Describe view and call move_forward.", end_of_turn=True)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                tg.create_task(self.prompt_loop())
                tg.create_task(self.safety_stop_loop())
                # tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            self.audio_stream.close()
            traceback.print_exception(EG)


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