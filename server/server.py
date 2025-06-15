import os
import logging
import numpy as np
import time
from typing import List, Optional, Dict, Tuple, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Response
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from tensorflow import keras
from datetime import datetime
import tempfile
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
import re
from sqlalchemy import func
import uuid
from fastapi import Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, DateTime, Table, Integer, ForeignKey, select
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from sqlalchemy.orm import sessionmaker, Session, relationship, declarative_base
from dotenv import load_dotenv
import pickle

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from he.fhe_provider import FHEProvider

load_dotenv(dotenv_path='.env.server')

fhe_provider = FHEProvider.generate()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is missing")
global_vars_runtime = {
    'last_aggregation_timestamp': 0,
    'latest_version': 0,
    'last_checked_timestamp': 0
}

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

client_model_association = Table(
    'client_model_association', Base.metadata,
    Column('client_id', String, ForeignKey('clients.client_id')),
    Column('model_id', Integer, ForeignKey('global_models.id'))
)

class Client(Base):
    __tablename__ = "clients"
    csn = Column(String, primary_key=True)
    client_id = Column(String, unique=True, nullable=False)
    api_key = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="Inactive")
    contribution_count = Column(Integer, default=0)
    models_contributed = relationship("GlobalModel", secondary=client_model_association, back_populates="clients")

class GlobalModel(Base):
    __tablename__ = "global_models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Integer, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    num_clients_contributed = Column(Integer, default=0)
    client_ids = Column(String)
    clients = relationship("Client", secondary=client_model_association, back_populates="models_contributed")

class GlobalAggregation(Base):
    __tablename__ = "global_aggregation"
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logging.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler()]
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logging.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")

    async def disconnect(self, client_id: str):
        for attempt in range(3):
            try:
                if client_id in self.active_connections:
                    del self.active_connections[client_id]
                    logging.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} - Error disconnecting client {client_id}: {e}")
                if attempt < 2:
                    time.sleep(2)

    async def broadcast_model_update(self, message: str):
        disconnected_clients = []
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(message)
                logging.info(f"Update notification sent to client {client_id}")
            except Exception as e:
                logging.error(f"Failed to send update to client {client_id}: {e}")
                disconnected_clients.append(client_id)
        for client_id in disconnected_clients:
            await self.disconnect(client_id)

CLIENT_ACCOUNT_URL = os.getenv("CLIENT_ACCOUNT_URL")
SERVER_ACCOUNT_URL = os.getenv("SERVER_ACCOUNT_URL")
CLIENT_CONTAINER_NAME = os.getenv("CLIENT_CONTAINER_NAME")
SERVER_CONTAINER_NAME = os.getenv("SERVER_CONTAINER_NAME")
LOCAL_CONTAINER_URL = os.getenv("LOCAL_CONTAINER_URL")
GLOBAL_CONTAINER_URL = os.getenv("GLOBAL_CONTAINER_URL")
CLIENT_CONTAINER_SAS_TOKEN = os.getenv("CLIENT_CONTAINER_SAS_TOKEN")
ARCH_BLOB_NAME = "model_architecture.h5"
CLIENT_NOTIFICATION_URL = os.getenv("CLIENT_NOTIFICATION_URL")

if not CLIENT_ACCOUNT_URL or not SERVER_ACCOUNT_URL:
    logging.error("SAS url environment variable is missing.")
    raise ValueError("Missing required environment variable: SAS url")

try:
    blob_service_client_client = BlobServiceClient(account_url=CLIENT_ACCOUNT_URL)
    blob_service_client_server = BlobServiceClient(account_url=SERVER_ACCOUNT_URL)
except Exception as e:
    logging.error(f"Failed to initialize Azure Blob Service: {e}")
    raise

def get_model_architecture() -> Optional[object]:
    try:
        container_client = blob_service_client_client.get_container_client(CLIENT_CONTAINER_NAME)
        blob_client = container_client.get_blob_client(ARCH_BLOB_NAME)
        arch_data = blob_client.download_blob().readall()
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            temp_file.write(arch_data)
            temp_path = temp_file.name
        model = keras.models.load_model(temp_path, compile=False)
        model.summary()
        os.unlink(temp_path)
        return model
    except Exception as e:
        logging.error(f"Error loading model architecture: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None

def load_weights_from_blob(
    blob_service_client: BlobServiceClient,
    container_name: str,
    last_aggregation_timestamp: int,
    encryption_type="fhe"
) -> Optional[Tuple[List[Tuple[str, List[Dict[str, Any]], Tuple]], List[int], List[float], int]]:
    try:
        pattern = re.compile(r"localweights/client([0-9a-fA-F\-]+)_v\d+_(\d{8}_\d{6})\.pkl")
        container_client = ContainerClient.from_container_url(LOCAL_CONTAINER_URL, credential=CLIENT_CONTAINER_SAS_TOKEN)
        weights_list = []
        num_examples_list = []
        loss_list = []
        new_last_aggregation_timestamp = last_aggregation_timestamp

        blobs = list(container_client.list_blobs())
        for blob in blobs:
            logging.info(f"Processing blob: {blob.name}")
            match = pattern.match(blob.name)
            if match:
                client_id = match.group(1)
                timestamp_str = match.group(2)
                timestamp_int = int(timestamp_str.replace("_", ""))
                if timestamp_int > last_aggregation_timestamp:
                    blob_client = container_client.get_blob_client(blob.name)
                    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
                        download_stream = blob_client.download_blob()
                        temp_file.write(download_stream.readall())
                        temp_path = temp_file.name
                    with open(temp_path, "rb") as f:
                        encrypted_weights = pickle.load(f)
                    blob_metadata = blob_client.get_blob_properties().metadata
                    if blob_metadata:
                        num_examples = int(blob_metadata.get('num_examples', 0))
                        loss = float(blob_metadata.get('loss', 0.0))
                        if num_examples == 0:
                            continue
                        num_examples_list.append(num_examples)
                        loss_list.append(loss)
                    os.unlink(temp_path)
                    weights_list.append((client_id, encrypted_weights))
                    new_last_aggregation_timestamp = max(new_last_aggregation_timestamp, timestamp_int)
            else:
                logging.warning(f"Blob name does not match pattern: {blob.name}")

        if not weights_list:
            logging.info(f"No new weights found since {last_aggregation_timestamp}.")
            return None, [], [], last_aggregation_timestamp

        logging.info(f"Loaded weights from {len(weights_list)} files.")
        return weights_list, num_examples_list, loss_list, new_last_aggregation_timestamp
    except Exception as e:
        logging.error(f"Error loading weights: {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        return None, [], [], last_aggregation_timestamp

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

def save_weights_to_blob(encrypted_weights: List[List[Dict[str, Any]]], filename: str) -> bool:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
            with open(temp_path, "wb") as f:
                pickle.dump(encrypted_weights, f)
        blob_client = blob_service_client_server.get_blob_client(container=SERVER_CONTAINER_NAME, blob=filename)
        with open(temp_path, "rb") as file:
            blob_client.upload_blob(file, overwrite=True)
        logging.info(f"Successfully saved weights to blob: {filename}")
        return True
    except Exception as e:
        logging.error(f"Error saving weights to blob: {e}")
        return False
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def federated_weighted_averaging(encrypted_weights_list: List[List[Dict[str, Any]]], num_examples_list: List[int], 
                                loss_list: List[float], provider, alpha: float = 0.5) -> List[Dict[str, Any]]:
    total_examples = sum(num_examples_list)
    if total_examples == 0:
        logging.error("Total examples is zero.")
        return None

    loss_weights = np.exp(-np.array(loss_list))
    loss_weights = loss_weights / np.sum(loss_weights)
    final_weights = []
    for i in range(len(encrypted_weights_list)):
        data_weight = num_examples_list[i] / total_examples
        combined_weight = alpha * data_weight + (1 - alpha) * loss_weights[i]
        final_weights.append(combined_weight)
    final_weights = np.array(final_weights) / np.sum(final_weights)
    return provider.secure_weighted_sum(encrypted_weights_list, final_weights)

def get_versioned_filename(version: int, prefix="g", extension=".pkl"):
    filename = f"{prefix}{version}.{extension}"
    return filename

def verify_admin(api_key: str):
    admin_key = os.getenv("ADMIN_API_KEY", "your_admin_secret_key")
    if api_key != admin_key:
        raise HTTPException(status_code=403, detail="Unauthorized admin access")

async def aggregate_weights_core(db: Session):
    try:
        global_vars_runtime['last_checked_timestamp'] = datetime.now().strftime("%Y%m%d%H%M%S")
        last_aggregation_timestamp = load_last_aggregation_timestamp(db)
        global_vars_runtime['last_aggregation_timestamp'] = last_aggregation_timestamp or 0
        logging.info(f"Loaded last processed timestamp: {global_vars_runtime['last_aggregation_timestamp']}")

        model = get_model_architecture()
        if not model:
            logging.critical("Failed to load model architecture")
            raise HTTPException(status_code=500, detail="Failed to load model architecture")

        weights_list_with_ids, num_examples_list, loss_list, new_timestamp = load_weights_from_blob(
            blob_service_client_client, 
            CLIENT_CONTAINER_NAME, 
            global_vars_runtime['last_aggregation_timestamp'],
            encryption_type="fhe"
        )

        if not weights_list_with_ids:
            logging.info("No new weights found in the blob")
            return {"status": "no_update", "message": "No new weights found", "num_clients": 0}
        if len(weights_list_with_ids) < 2:
            logging.info("Insufficient weights for aggregation")
            return {"status": "no_update", "message": "Only 1 weight file found", "num_clients": 1}
        if not num_examples_list:
            logging.error("Example counts are missing")
            return {"status": "error", "message": "Example counts missing for aggregation"}

        latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
        global_vars_runtime['latest_version'] = latest_model.version if latest_model else 0
        logging.info(f"Latest model version loaded: {global_vars_runtime['latest_version']}")

        global_vars_runtime['latest_version'] += 1
        if db.query(GlobalModel).filter_by(version=global_vars_runtime['latest_version']).first():
            logging.error(f"Duplicate model version detected: {global_vars_runtime['latest_version']}")
            raise HTTPException(status_code=409, detail=f"Model with version {global_vars_runtime['latest_version']} already exists")

        filename = get_versioned_filename(global_vars_runtime['latest_version'])
        logging.info(f"Preparing to save aggregated weights as: {filename}")

        encrypted_weights_list = [weights for _, weights in weights_list_with_ids]
        logging.info(f"Aggregating weights from {len(encrypted_weights_list)} clients")
        avg_encrypted_weights = federated_weighted_averaging(encrypted_weights_list, num_examples_list, loss_list, fhe_provider)
        logging.info("Aggregation completed successfully.")

        if not avg_encrypted_weights or not save_weights_to_blob(avg_encrypted_weights, filename):
            logging.critical("Failed to save aggregated weights to blob")
            raise HTTPException(status_code=500, detail="Failed to save aggregated weights")

        save_last_aggregation_timestamp(db, new_timestamp)
        logging.info(f"New timestamp saved to the database: {new_timestamp}")

        contributing_client_ids = [id for id, _ in weights_list_with_ids]
        new_model = GlobalModel(
            version=global_vars_runtime['latest_version'],
            num_clients_contributed=len(encrypted_weights_list),
            client_ids=",".join(contributing_client_ids)
        )
        db.add(new_model)
        db.query(Client).filter(Client.client_id.in_(contributing_client_ids)).update(
            {"contribution_count": Client.contribution_count + 1},
            synchronize_session=False
        )
        db.commit()
        logging.info(f"Model version {global_vars_runtime['latest_version']} saved and database updated")

        await manager.broadcast_model_update(f"NEW_MODEL:{filename}")
        logging.info(f"Clients notified of new model: {filename}")

        return {
            "status": "success",
            "message": f"Aggregated weights saved as {filename}",
            "num_clients": len(encrypted_weights_list)
        }
    except SQLAlchemyError as db_error:
        logging.error(f"Database error during aggregation: {db_error}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Database error occurred")
    except HTTPException as http_exc:
        logging.error(f"HTTP Exception: {http_exc.detail}")
        db.rollback()
        raise
    except Exception as e:
        logging.exception("Unexpected error during aggregation")
        db.rollback()
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

manager = ConnectionManager()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "HELLO, WORLD. Welcome to the SecureFL Server!"}

@app.get("/health", response_class=JSONResponse)
async def health_check():
    return {"status": "healthy"}

@app.head("/health")
async def health_check_monitor():
    return Response(status_code=200)

@app.get("/get_data")
async def get_all_data(db: Session = Depends(get_db)):
    try:
        clients = db.execute(select(Client)).scalars().all()
        global_models = db.execute(select(GlobalModel)).scalars().all()
        global_vars_table = db.execute(select(GlobalAggregation)).scalars().all()
        return {
            "clients": clients,
            "global_models": global_models,
            "global_aggregation": global_vars_table,
            "last_checked_timestamp": global_vars_runtime['last_checked_timestamp']
        }
    except Exception as e:
        logging.error(f"Error in /get_data endpoint: {e}")
        return {"error": "Failed to fetch data. Please try again later."}

@app.post("/register")
async def register(
    csn: str = Body(..., embed=True),
    admin_api_key: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    try:
        verify_admin(admin_api_key)
        existing_client = db.query(Client).filter(Client.csn == csn).first()
        if existing_client:
            raise HTTPException(status_code=400, detail="Client already registered")
        client_id = str(uuid.uuid4())
        api_key = str(uuid.uuid4())
        new_client = Client(csn=csn, client_id=client_id, api_key=api_key)
        db.add(new_client)
        db.commit()
        return {
            "status": "success",
            "message": "Client registered successfully",
            "data": {"client_id": client_id, "api_key": api_key}
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.error(f"Error during client registration: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the registration")

@app.get("/aggregate-weights")
async def aggregate_weights(db: Session = Depends(get_db)):
    try:
        return await aggregate_weights_core(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, db: Session = Depends(get_db)):
    retry_attempts = 3
    try:
        client = db.query(Client).filter(Client.client_id == client_id).first()
        if not client:
            logging.warning(f"Client {client_id} not found in database. Closing WebSocket.")
            await websocket.close(code=1008, reason="Unauthorized")
            return
        logging.info(f"Client {client_id} found in DB: {client}")

        await manager.connect(client_id, websocket)
        for attempt in range(retry_attempts):
            try:
                client.status = "Active"
                db.commit()
                logging.info(f"Client {client_id} connected successfully, status updated to 'Active'.")
                break
            except SQLAlchemyError as db_error:
                db.rollback()
                logging.error(f"Attempt {attempt + 1} - Failed to update 'Active' status for {client_id}: {db_error}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)
                else:
                    raise

        await websocket.send_text(f"Your status is now: {client.status}")
        latest_model = db.query(GlobalModel).order_by(GlobalModel.version.desc()).first()
        global_vars_runtime['latest_version'] = latest_model.version if latest_model else 0
        latest_model_version = f"g{global_vars_runtime['latest_version']}.pkl"
        await websocket.send_text(f"LATEST_MODEL:{latest_model_version}")

        while True:
            try:
                data = await websocket.receive_text()
                if not data:
                    break
                for attempt in range(retry_attempts):
                    try:
                        db.refresh(client)
                        if client.status != "Active":
                            client.status = "Active"
                            db.commit()
                            await websocket.send_text(f"Your updated status is: {client.status}")
                        break
                    except SQLAlchemyError as db_error:
                        db.rollback()
                        logging.error(f"Attempt {attempt + 1} - Database error for client {client_id}: {db_error}")
                        if attempt < retry_attempts - 1:
                            await asyncio.sleep(2)
                        else:
                            raise
            except WebSocketDisconnect:
                logging.info(f"Client {client_id} disconnected gracefully.")
                break
            except Exception as e:
                logging.error(f"Error handling message from client {client_id}: {e}")
                await websocket.send_text("An error occurred. Please try again later.")
                break
    except SQLAlchemyError as db_error:
        logging.error(f"Database error for client {client_id}: {db_error}")
        db.rollback()
        await websocket.close(code=1002, reason="Database error.")
    except WebSocketDisconnect:
        logging.info(f"Client {client_id} disconnected unexpectedly.")
    except Exception as e:
        logging.error(f"Unexpected error for client {client_id}: {e}")
        await websocket.close(code=1000, reason="Internal server error.")
    finally:
        for attempt in range(retry_attempts):
            try:
                await manager.disconnect(client_id)
                break
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} - Failed to disconnect client {client_id}: {e}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)
        for attempt in range(retry_attempts):
            try:
                if client:
                    db.refresh(client)
                    client.status = "Inactive"
                    db.commit()
                    logging.info(f"Client {client_id} is now inactive. DB updated successfully.")
                else:
                    logging.warning(f"Skipping status update: Client {client_id} does not exist.")
                break
            except SQLAlchemyError as db_error:
                db.rollback()
                logging.error(f"Attempt {attempt + 1} - Failed to update 'Inactive' status for {client_id}: {db_error}")
                if attempt < retry_attempts - 1:
                    await asyncio.sleep(2)
                else:
                    raise
        logging.info(f"Cleanup completed for client {client_id}.")

scheduler = BackgroundScheduler()

@scheduler.scheduled_job(CronTrigger(minute="*/5"))
def scheduled_aggregate_weights():
    logging.info("Scheduled task: Starting weight aggregation process.")
    db = SessionLocal()
    try:
        asyncio.run(aggregate_weights_core(db))
    except Exception as e:
        logging.error(f"Error during scheduled weight aggregation: {e}")
    finally:
        db.close()
scheduler.start()

if __name__ == "__main__":
    import uvicorn
    logging.info("Starting Server...")
    uvicorn.run(app, host="localhost", port=8000)