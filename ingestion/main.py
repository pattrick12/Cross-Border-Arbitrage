import os
from dotenv import load_dotenv
import requests
import ujson
from confluent_kafka import Producer
import sys

# Add parent dir to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

load_dotenv()
logger = get_logger("Ingestion")

import time
import random
import json
from datetime import datetime

class MockStreamGenerator:
    def __init__(self, topic="ticks", broker="localhost:19092"):
        self.topic = topic
        self.producer = Producer({'bootstrap.servers': broker})
        self.pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD"]
        self.prices = {
            "EUR_USD": 1.0500,
            "USD_JPY": 150.00,
            "GBP_USD": 1.2500,
            "AUD_USD": 0.6500,
            "USD_CAD": 1.3500
        }

    def generate_tick(self):
        pair = random.choice(self.pairs)
        # Random walk
        change = random.uniform(-0.0005, 0.0005)
        self.prices[pair] += change
        
        # Create tick structure matching OANDA format roughly
        tick = {
            "type": "PRICE",
            "time": datetime.utcnow().isoformat() + "Z",
            "instrument": pair,
            "bids": [{"price": str(round(self.prices[pair] - 0.0001, 5)), "liquidity": 1000000}],
            "asks": [{"price": str(round(self.prices[pair] + 0.0001, 5)), "liquidity": 1000000}],
            "closeoutBid": str(round(self.prices[pair] - 0.0001, 5)),
            "closeoutAsk": str(round(self.prices[pair] + 0.0001, 5)),
            "status": "tradeable",
            "tradeable": True
        }
        return tick

    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            # logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")
            pass

    def run(self):
        logger.info(f"Starting Mock Stream to topic '{self.topic}'...")
        try:
            while True:
                tick = self.generate_tick()
                self.producer.produce(
                    self.topic,
                    key=tick["instrument"],
                    value=ujson.dumps(tick),
                    callback=self.delivery_report
                )
                self.producer.poll(0)
                # Simulate high frequency: 10ms - 100ms delay
                time.sleep(random.uniform(0.01, 0.1))
        except KeyboardInterrupt:
            logger.info("Stopping Mock Stream...")
        finally:
            self.producer.flush()

def main():
    logger.info("Starting OANDA Ingestion Service (MOCK MODE)...")
    generator = MockStreamGenerator()
    generator.run()

if __name__ == "__main__":
    main()