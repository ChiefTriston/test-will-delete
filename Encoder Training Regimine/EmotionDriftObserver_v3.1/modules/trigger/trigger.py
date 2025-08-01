# modules/trigger/trigger.py

import os
import time
import signal
import uuid
import queue
import portalocker
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Shared job queue for incoming jobs
job_queue = queue.Queue()

class ReadyHandler(FileSystemEventHandler):
    def __init__(self, config):
        self.config = config
        self.raw_dir = config['raw_audio_dir']

    def on_created(self, event):
        # Trigger on .ready files
        if event.src_path.endswith('.ready'):
            wav_path = event.src_path[:-6] + '.wav'
            if os.path.exists(wav_path):
                with open(event.src_path, 'r+') as lock_file:
                    portalocker.lock(lock_file, portalocker.LOCK_EX)
                    job_id = str(uuid.uuid4())
                    job_queue.put((self.config, job_id, wav_path))
                    lock_file.truncate(0)
                    portalocker.unlock(lock_file)
                os.remove(event.src_path)

def run_trigger_watcher(config):
    handler = ReadyHandler(config)
    observer = Observer()
    observer.schedule(handler, config['raw_audio_dir'], recursive=False)
    observer.start()

    def shutdown(signum, frame):
        observer.stop()
        observer.join()
        print("Watcher stopped gracefully.")
        exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()