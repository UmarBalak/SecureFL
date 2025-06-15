import logging
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from models.database import GlobalAggregation, GlobalModel, Client
from services.fhe_service import fhe_service

class AggregationService:
    @staticmethod
    def load_last_aggregation_timestamp(db: Session) -> int:
        for attempt in range(3):
            try:
                timestamp = db.query(GlobalAggregation).filter_by(key="last_aggregation_timestamp").first()
                return int(timestamp.value) if timestamp else 0
            except OperationalError as db_error:
                logging.error(f"Attempt {attempt + 1} - Database error: {db_error}")
                db.rollback()
                if attempt < 2:
                    time.sleep(2)
                else:
                    raise
    
    @staticmethod
    def save_last_aggregation_timestamp(db: Session, new_timestamp):
        try:
            timestamp_record = db.query(GlobalAggregation).filter_by(key="last_aggregation_timestamp").first()
            if timestamp_record:
                timestamp_record.value = new_timestamp
            else:
                new_record = GlobalAggregation(key="last_aggregation_timestamp", value=new_timestamp)
                db.add(new_record)
            db.commit()
        except Exception as e:
            logging.error(f"Error saving last aggregation timestamp: {e}")
            raise
    
    @staticmethod
    def federated_averaging(
        encrypted_weights_list: List[List[Tuple[List[Dict[str, Any]], Tuple]]], 
        num_examples_list: List[int]
    ) -> List[Dict[str, Any]]:
        num_clients = len(encrypted_weights_list)
        if num_clients == 0:
            logging.error("No encrypted weights provided for aggregation.")
            return None
        
        # Validate input lengths
        if len(encrypted_weights_list) != len(num_examples_list):
            logging.error("Mismatched lengths: weights and examples")
            return None
        
        # Validate tensor shapes
        if encrypted_weights_list:
            expected_length = len(encrypted_weights_list[0][0])
            for weights, _ in encrypted_weights_list:
                if len(weights) != expected_length:
                    logging.error("Inconsistent tensor lengths in encrypted weights")
                    return None
        
        # Simple FedAvg: equal weights for all clients
        weights = np.array([1.0 / num_clients] * num_clients)
        logging.info(f"Performing FedAvg with {num_clients} clients, each with weight {1.0 / num_clients}")
        
        return fhe_service.provider.secure_weighted_sum(encrypted_weights_list, weights)
    
    @staticmethod
    def get_versioned_filename(version: int, prefix="g", extension=".pkl"):
        return f"{prefix}{version}.{extension}"

aggregation_service = AggregationService()