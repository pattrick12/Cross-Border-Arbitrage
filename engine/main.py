import os
import sys
import json
import time
import uuid
import threading
import queue
import math
from confluent_kafka import Consumer, Producer, KafkaError
from questdb.ingress import Sender, TimestampNanos
from dotenv import load_dotenv

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

# Import C++ Extension
import graph_core

load_dotenv()
logger = get_logger("ArbitrageEngine")

# Global signal queue (thread-safe)
signal_queue = queue.Queue(maxsize=1000)

# Expanded currency list (15 currencies)
CURRENCIES = [
    # Major currencies
    "EUR", "USD", "JPY", "GBP", "AUD", "CAD", "CHF",
    "INR",  # Indian Rupee
    "CNY",  # Chinese Yuan
    "RUB",  # Russian Ruble
    "KWD",  # Kuwaiti Dinar
    "SGD",  # Singapore Dollar
    "HKD",  # Hong Kong Dollar
    "BRL",  # Brazilian Real
    "MXN",  # Mexican Peso
]


def signal_storage_worker():
    """
    Background thread that consumes signals from queue and writes to:
    1. Redpanda (arbitrage_signals topic)
    2. QuestDB (arbitrage_signals table)
    """
    # Kafka Producer for signals
    producer_conf = {'bootstrap.servers': 'localhost:19092'}
    producer = Producer(producer_conf)
    
    # QuestDB Sender
    qdb_conf = 'tcp::addr=localhost:9009;'
    
    try:
        with Sender.from_conf(qdb_conf) as sender:
            while True:
                try:
                    # Block until signal available (timeout 1s to allow clean shutdown)
                    signal_data = signal_queue.get(timeout=1.0)
                    
                    if signal_data is None:  # Poison pill for shutdown
                        break
                    
                    cycle = signal_data['cycle']
                    detected_at = signal_data['timestamp']
                    profit_pct = signal_data['profit_pct']
                    order = signal_data['order']
                    
                    # Generate unique ID
                    signal_id = str(uuid.uuid4())[:8]
                    
                    # Format cycle as string
                    path_str = ' -> '.join(cycle)
                    
                    # Publish to Redpanda
                    message = {
                        'signal_id': signal_id,
                        'path': path_str,
                        'currencies': order,
                        'profit_pct': profit_pct,
                        'detected_at': detected_at,
                        'source': 'MOCK'
                    }
                    
                    producer.produce(
                        'arbitrage_signals',
                        key=signal_id,
                        value=json.dumps(message)
                    )
                    producer.poll(0)  # Non-blocking
                    
                    # Write to QuestDB
                    sender.row(
                        'arbitrage_signals',
                        symbols={
                            'signal_id': signal_id,
                            'source': 'MOCK'
                        },
                        columns={
                            'path': path_str,
                            'currencies': order,
                            'profit_pct': profit_pct
                        },
                        at=TimestampNanos.now()
                    )
                    sender.flush()
                    
                    logger.info(f"Stored signal {signal_id}: {path_str} ({profit_pct:.4f}%) [Order: {order}]")
                    
                except queue.Empty:
                    continue  # No signals, keep waiting
                    
    except Exception as e:
        logger.error(f"Signal storage worker error: {e}")
    finally:
        producer.flush()
        logger.info("Signal storage worker stopped")


def main():
    logger.info("Starting Arbitrage Engine (C++ Core)...")
    logger.info(f"Supported currencies: {CURRENCIES}")

    # Start signal storage worker thread
    storage_thread = threading.Thread(target=signal_storage_worker, daemon=True)
    storage_thread.start()
    logger.info("Signal storage worker started")

    # Initialize Graph with expanded currency list
    graph = graph_core.ArbitrageGraph(CURRENCIES)
    logger.info(f"Initialized Graph with {len(CURRENCIES)} currencies")

    # Kafka Consumer Config
    conf = {
        'bootstrap.servers': 'localhost:19092',
        'group.id': 'arb-engine-group',
        'auto.offset.reset': 'latest'
    }
    consumer = Consumer(conf)
    consumer.subscribe(['ticks'])

    logger.info("Listening for ticks...")

    try:
        while True:
            msg = consumer.poll(0.1)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(msg.error())
                    break

            try:
                data = json.loads(msg.value().decode('utf-8'))
                instrument = data.get('instrument')
                
                if instrument and '_' in instrument:
                    base, quote = instrument.split('_')
                    
                    # OANDA prices
                    bid = float(data['closeoutBid'])
                    ask = float(data['closeoutAsk'])
                    
                    if bid > 0 and ask > 0:
                        # Update Base -> Quote (Sell Base)
                        graph.update_rate(base, quote, bid)
                        
                        # Update Quote -> Base (Buy Base)
                        graph.update_rate(quote, base, 1.0 / ask)
                        
                        # Detect Arbitrage with profit calculation (HOT PATH)
                        result = graph.detect_arbitrage_with_profit()
                        
                        if result.cycle:
                            # NON-BLOCKING: Just queue the signal with profit info
                            try:
                                signal_queue.put_nowait({
                                    'cycle': result.cycle,
                                    'profit_pct': result.profit_pct,
                                    'order': result.order,
                                    'timestamp': time.time_ns()
                                })
                                logger.info(f"ARBITRAGE DETECTED: {result.cycle} | Profit: {result.profit_pct:.4f}% | Order: {result.order}")
                            except queue.Full:
                                logger.warning("Signal queue full, dropping signal")
                        
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Stopping Arbitrage Engine...")
        # Send poison pill to worker thread
        signal_queue.put(None)
        storage_thread.join(timeout=2.0)
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
