import os
import sys
import json
from confluent_kafka import Consumer, KafkaError
from questdb.ingress import Sender, IngressError, TimestampNanos
from dotenv import load_dotenv
import time

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

load_dotenv()
logger = get_logger("Storage")

def main():
    logger.info("Starting Storage Service (QuestDB Ingestion)...")

    # Kafka Consumer Config
    conf = {
        'bootstrap.servers': 'localhost:19092',
        'group.id': 'storage-group',
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(conf)
    consumer.subscribe(['ticks'])

    # QuestDB Sender Config
    qdb_host = 'localhost'
    qdb_port = 9009
    
    qdb_conf = f'tcp::addr={qdb_host}:{qdb_port};'
    logger.info(f"Connecting to QuestDB with conf: {qdb_conf}")

    try:
        with Sender.from_conf(qdb_conf) as sender:
            while True:
                msg = consumer.poll(1.0)

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
                    # Expected data:
                    # {
                    #     "type": "PRICE",
                    #     "time": "2023-10-27T10:00:00.123456Z",
                    #     "instrument": "EUR_USD",
                    #     "bids": [{"price": "1.0500", ...}],
                    #     "asks": [{"price": "1.0501", ...}],
                    #     ...
                    # }
                    
                    if data.get('type') == 'PRICE':
                        instrument = data.get('instrument')
                        bid = float(data['bids'][0]['price'])
                        ask = float(data['asks'][0]['price'])
                        # Parse time or use current time?
                        # QuestDB ILP usually takes timestamp in nanos.
                        # For simplicity, let's use current time if parsing is complex,
                        # but parsing is better for historical accuracy.
                        # "time": "2025-11-21T02:02:31.123456Z"
                        # We can use TimestampNanos.now() for simplicity in Phase 1.
                        
                        sender.row(
                            'forex_ticks',
                            symbols={
                                'symbol': instrument,
                                'source': 'MOCK'
                            },
                            columns={
                                'bid': bid,
                                'ask': ask,
                                'size_bid': float(data['bids'][0]['liquidity']),
                                'size_ask': float(data['asks'][0]['liquidity'])
                            },
                            at=TimestampNanos.now()
                        )
                        
                        # Flush periodically or let Sender handle it (auto_flush is default?)
                        # Sender buffers. We should flush explicitly or configure it.
                        # For high throughput, auto flush is good.
                        sender.flush()
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

    except KeyboardInterrupt:
        logger.info("Stopping Storage Service...")
    except IngressError as e:
        logger.error(f"QuestDB Ingress Error: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    main()
