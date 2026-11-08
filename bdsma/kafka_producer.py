"""Optional Kafka producer for an exported BDSMA events JSONL file.

Install kafka-python, start a Kafka broker, then run:
    python bdsma/kafka_producer.py data/bdsma_events.jsonl --bootstrap localhost:9092
"""
import argparse
import json
import time
from kafka import KafkaProducer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("events_file")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default="supply-chain-events")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        retries=3,
    )
    sent = 0
    try:
        with open(args.events_file, "r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                producer.send(args.topic, json.loads(line))
                sent += 1
                time.sleep(max(args.delay, 0))
        producer.flush()
    finally:
        producer.close()
    print(f"Published {sent} events to topic '{args.topic}'.")


if __name__ == "__main__":
    main()
