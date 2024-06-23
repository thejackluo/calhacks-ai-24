import asyncio
import base64
import json
import tempfile
import logging
import io
import wave
import numpy as np
import websockets
import soundfile
from playsound import playsound
from pyaudio import Stream as PyAudioStream
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=1)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.DEBUG
)

class Connection:

    @classmethod
    async def connect(
        cls,
        socket_url: str,
        audio_stream: PyAudioStream,
        sample_rate: int,
        sample_width: int,
        num_channels: int,
        chunk_size: int,
    ):
        while True:
            try:
                async with websockets.connect(socket_url) as socket:
                    print("Connected to WebSocket")
                    send_task = asyncio.create_task(
                        cls._send_audio_data(
                            socket,
                            audio_stream,
                            sample_rate,
                            sample_width,
                            num_channels,
                            chunk_size,
                        )
                    )
                    receive_task = asyncio.create_task(cls._receive_audio_data(socket))
                    await asyncio.gather(receive_task, send_task)
            except websockets.exceptions.ConnectionClosed:
                print(
                    "WebSocket connection closed. Attempting to reconnect in 5 seconds..."
                )
                await asyncio.sleep(5)
            except Exception as e:
                print(
                    f"An error occurred: {e}. Attempting to reconnect in 5 seconds..."
                )
                await asyncio.sleep(5)

    @classmethod
    async def _receive_audio_data(cls, socket):
        try:
            async for message in socket:
                try:
                    json_message = json.loads(message)
                    print("Received JSON message:", json_message)

                    if json_message.get("type") == "audio_output":
                        audio_data = base64.b64decode(json_message["data"])

                        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
                            tmpfile.write(audio_data)
                            tmpfile.flush()
                            playsound(tmpfile.name)
                            print("Audio played")

                except ValueError as e:
                    print(f"Failed to parse JSON, error: {e}")
                except KeyError as e:
                    print(f"Key error in JSON data: {e}")

        except Exception as e:
            print(f"An error occurred while receiving audio: {e}")

    @classmethod
    async def _read_audio_stream_non_blocking(cls, audio_stream, chunk_size):
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            executor, audio_stream.read, chunk_size, False
        )
        return data

    @classmethod
    async def _send_audio_data(
        cls,
        socket,
        audio_stream: PyAudioStream,
        sample_rate: int,
        sample_width: int,
        num_channels: int,
        chunk_size: int,
    ):
        wav_buffer = io.BytesIO()
        headers_sent = False

        while True:
            data = await cls._read_audio_stream_non_blocking(audio_stream, chunk_size)
            if num_channels == 2:
                stereo_data = np.frombuffer(data, dtype=np.int16)
                mono_data = ((stereo_data[0::2] + stereo_data[1::2]) / 2).astype(np.int16)
                data = mono_data.tobytes()

            np_array = np.frombuffer(data, dtype="int16")
            soundfile.write(
                wav_buffer,
                np_array,
                samplerate=sample_rate,
                subtype="PCM_16",
                format="RAW",
            )

            wav_content = wav_buffer.getvalue()
            if not headers_sent:
                header_buffer = io.BytesIO()
                with wave.open(header_buffer, "wb") as wf:
                    wf.setnchannels(num_channels)
                    wf.setsampwidth(sample_width)
                    wf.setframerate(sample_rate)
                    wf.setnframes(chunk_size)

                    wf.writeframes(b"")

                headers = header_buffer.getvalue()
                wav_content = headers + wav_content
                headers_sent = True

            encoded_audio = base64.b64encode(wav_content).decode('utf-8')
            json_message = json.dumps({"type": "audio_input", "data": encoded_audio})
            await socket.send(json_message)

            wav_buffer = io.BytesIO()
