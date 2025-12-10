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
        
        # Expanded currency pairs (covering all 15 currencies)
        self.pairs = [
            # Major pairs (USD-based)
            "EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_CAD", "USD_CHF",
            # Cross pairs
            "EUR_JPY", "EUR_GBP", "GBP_JPY", "EUR_CHF", "AUD_JPY",
            # Emerging market pairs
            "USD_INR", "USD_CNY", "USD_RUB", "USD_KWD",
            "USD_SGD", "USD_HKD", "USD_BRL", "USD_MXN",
            # Cross with emerging
            "EUR_INR", "EUR_CNY", "GBP_INR", "JPY_INR",
            "EUR_SGD", "GBP_HKD", "AUD_SGD",
        ]
        
        # Realistic base prices (as of late 2024)
        self.prices = {
            # Major pairs
            "EUR_USD": 1.0550,
            "USD_JPY": 149.50,
            "GBP_USD": 1.2650,
            "AUD_USD": 0.6550,
            "USD_CAD": 1.3550,
            "USD_CHF": 0.8750,
            # Cross pairs
            "EUR_JPY": 157.70,
            "EUR_GBP": 0.8340,
            "GBP_JPY": 189.10,
            "EUR_CHF": 0.9230,
            "AUD_JPY": 97.90,
            # Emerging markets (vs USD)
            "USD_INR": 83.25,
            "USD_CNY": 7.24,
            "USD_RUB": 92.50,
            "USD_KWD": 0.3075,  
            "USD_SGD": 1.3450,
            "USD_HKD": 7.8150,
            "USD_BRL": 4.9200,
            "USD_MXN": 17.25,
            # Cross with emerging
            "EUR_INR": 87.83,
            "EUR_CNY": 7.64,
            "GBP_INR": 105.30,
            "JPY_INR": 0.5570,
            "EUR_SGD": 1.4190,
            "GBP_HKD": 9.8850,
            "AUD_SGD": 0.8810,
        }
        
        # Volatility settings (different for each pair)
        self.volatility = {
            # Major pairs - lower volatility
            "EUR_USD": 0.0003, "USD_JPY": 0.05, "GBP_USD": 0.0003, 
            "AUD_USD": 0.0003, "USD_CAD": 0.0003, "USD_CHF": 0.0003,
            # Cross pairs - medium volatility
            "EUR_JPY": 0.08, "EUR_GBP": 0.0003, "GBP_JPY": 0.10,
            "EUR_CHF": 0.0003, "AUD_JPY": 0.05,
            # Emerging - higher volatility
            "USD_INR": 0.02, "USD_CNY": 0.005, "USD_RUB": 0.20,
            "USD_KWD": 0.0001, "USD_SGD": 0.001, "USD_HKD": 0.001,
            "USD_BRL": 0.015, "USD_MXN": 0.03,
            # Cross emerging
            "EUR_INR": 0.03, "EUR_CNY": 0.006, "GBP_INR": 0.04,
            "JPY_INR": 0.0003, "EUR_SGD": 0.001, "GBP_HKD": 0.002,
            "AUD_SGD": 0.001,
        }

    def generate_tick(self):
        pair = random.choice(self.pairs)
        volatility = self.volatility.get(pair, 0.0005)
        
        # Random walk with pair-specific volatility
        change = random.uniform(-volatility, volatility)
        self.prices[pair] += change
        
        # Ensure price stays positive
        self.prices[pair] = max(self.prices[pair], 0.0001)
        
        # Calculate spread based on pair type
        if pair in ["USD_KWD", "USD_HKD", "USD_CNY"]:
            spread = 0.00005  # Tight spread for stable pairs
        elif "RUB" in pair or "BRL" in pair:
            spread = 0.005  # Wider spread for volatile pairs
        else:
            spread = 0.0002  # Default spread
        
        # Create tick structure matching OANDA format
        tick = {
            "type": "PRICE",
            "time": datetime.utcnow().isoformat() + "Z",
            "instrument": pair,
            "bids": [{"price": str(round(self.prices[pair] - spread, 5)), "liquidity": 1000000}],
            "asks": [{"price": str(round(self.prices[pair] + spread, 5)), "liquidity": 1000000}],
            "closeoutBid": str(round(self.prices[pair] - spread, 5)),
            "closeoutAsk": str(round(self.prices[pair] + spread, 5)),
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
        logger.info(f"Generating ticks for {len(self.pairs)} currency pairs")
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