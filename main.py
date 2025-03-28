import ecdsa
import hashlib
import json
import random
import time
import threading
import os
import sys
import requests
from flask import Flask, request, jsonify

################################################
# Global Settings & Files
################################################
LEDGER_FILE = "ledger.json"

wallets = {}      # PublicKeyStr -> Wallet
peers = []        # List of peer node URLs
REGIONS = ["North", "South", "East", "West"]

# Depending on node mode:
validators = []   # For validator nodes only
mempool = []      # For validator nodes only

node_mode = "wallet"  # default, can be "wallet" or "validator"

################################################
# Classes
################################################
class Wallet:
    """
    Simple wallet class with a private/public key pair
    and a TRIX coin balance.
    """
    def __init__(self, region):
        self.region = region
        self.private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        self.public_key = self.private_key.get_verifying_key()
        self.balance = 100.0  # Starting TRIX

    def sign(self, data: str) -> str:
        signature = self.private_key.sign(data.encode())
        return signature.hex()

    def get_public_key_str(self) -> str:
        return self.public_key.to_string().hex()


class Validator:
    """
    Validator nodes hold a private/public key pair,
    plus a local balance for fees.
    """
    def __init__(self, id, region):
        self.id = id
        self.region = region
        self.load = 0
        self.private_key = ecdsa.SigningKey.generate(curve=ecdsa.SECP256k1)
        self.public_key = self.private_key.get_verifying_key()
        self.balance = 0.0

    def sign(self, data: str) -> str:
        return self.private_key.sign(data.encode()).hex()

    def verify_signature(self, data: str, signature: str, public_key_hex: str) -> bool:
        try:
            vk_bytes = bytes.fromhex(public_key_hex)
            vk = ecdsa.VerifyingKey.from_string(vk_bytes, curve=ecdsa.SECP256k1)
            return vk.verify(bytes.fromhex(signature), data.encode())
        except:
            return False

    def __str__(self):
        return f"Validator {self.id} in {self.region} (load: {self.load}, balance: {self.balance} TRIX)"


class Transaction:
    """
    Represents a single transaction (which becomes a 'block').
    """
    def __init__(self, sender, receiver, amount, fee, data, one_validator=False):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount
        self.fee = fee
        self.data = data
        self.one_validator_confirmation = one_validator

        self.sender_signature = None
        self.sender_validator_signature = None
        self.receiver_validator_signature = None
        self.transaction_hash = None
        self.fee_distribution = None
        self.status = "pending"

    def serialize(self) -> str:
        return json.dumps({
            "sender": self.sender,
            "receiver": self.receiver,
            "amount": self.amount,
            "fee": self.fee,
            "data": self.data,
            "one_validator_confirmation": self.one_validator_confirmation,
            "sender_signature": self.sender_signature,
            "sender_validator_signature": self.sender_validator_signature,
            "receiver_validator_signature": self.receiver_validator_signature,
            "fee_distribution": self.fee_distribution,
            "status": self.status
        }, sort_keys=True)

    def compute_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode()).hexdigest()


class Blockchain:
    """
    Simple blockchain ledger that stores confirmed transactions (blocks)
    in a local JSON file for persistence.
    """
    def __init__(self):
        self.chain = []  # list of block dicts
        self.load_from_file()

    def add_block(self, block_data: dict):
        self.chain.append(block_data)
        self.save_to_file()

    def save_to_file(self):
        try:
            with open(LEDGER_FILE, "w") as f:
                json.dump(self.chain, f, indent=4)
        except Exception as e:
            print("Error saving ledger:", e)

    def load_from_file(self):
        if os.path.exists(LEDGER_FILE):
            try:
                with open(LEDGER_FILE, "r") as f:
                    self.chain = json.load(f)
            except Exception as e:
                print("Error loading ledger:", e)
                self.chain = []
        else:
            self.chain = []

    def list_blocks(self):
        if not self.chain:
            print("No blocks in blockchain.")
        else:
            for idx, block in enumerate(self.chain):
                print(f"Block {idx+1}: {json.dumps(block)}")

# A single global blockchain
blockchain = Blockchain()

################################################
# Validator Node Utilities
################################################
def init_validators():
    """ For validator nodes only. Creates 2 validators per region. """
    global validators
    validator_id_counter = 1
    for region in REGIONS:
        for _ in range(2):
            validators.append(Validator(validator_id_counter, region))
            validator_id_counter += 1

def get_validator_for_region(region: str):
    """Choose the validator in the given region with the lowest load."""
    region_validators = [v for v in validators if v.region == region]
    if not region_validators:
        return None
    chosen = min(region_validators, key=lambda v: v.load)
    chosen.load += 1
    return chosen

def release_validator_load(validator: Validator):
    if validator.load > 0:
        validator.load -= 1

def broadcast_to_majority(tx_hash: str) -> bool:
    """Simulate broadcast to majority of validators for MVP."""
    if len(validators) == 0:
        # If for some reason no validators exist, fail
        return False
    confirmations = len(validators)
    return confirmations > (len(validators) / 2)

def final_consensus(sender_region: str, tx_hash: str) -> bool:
    """Simulate final consensus from a random validator in the sender region."""
    region_validators = [v for v in validators if v.region == sender_region]
    if not region_validators:
        return False
    chosen = random.choice(region_validators)
    print(f"Final consensus obtained from {chosen}")
    return True

################################################
# P2P Logic
################################################
app = Flask(__name__)

@app.route("/chain", methods=["GET"])
def get_chain():
    return jsonify(blockchain.chain), 200

@app.route("/peers", methods=["GET"])
def get_peers():
    return jsonify(peers), 200

@app.route("/new_block", methods=["POST"])
def receive_new_block():
    """
    Another node has forged a block and broadcast it.
    We add it if we don't have it already.
    """
    block_data = request.get_json()
    tx_hash = block_data.get("transaction_hash")
    # If we already have this block, skip
    for existing in blockchain.chain:
        if existing.get("transaction_hash") == tx_hash:
            return jsonify({"status": "exists"}), 200
    # Otherwise, add to chain
    blockchain.add_block(block_data)
    print(f"New block received (hash={tx_hash}). Chain size={len(blockchain.chain)}")
    return jsonify({"status": "success"}), 200

@app.route("/new_transaction", methods=["POST"])
def receive_new_transaction():
    """
    Wallet node is submitting a new transaction to us (validator).
    If we are a validator node, we put it in our mempool.
    """
    data = request.get_json()
    try:
        tx = Transaction(
            sender=data["sender"],
            receiver=data["receiver"],
            amount=float(data["amount"]),
            fee=float(data["fee"]),
            data=data["data"],
            one_validator=data.get("one_validator_confirmation", False)
        )
        if node_mode == "validator":
            mempool.append(tx)
            print("New transaction received & added to mempool.")
        else:
            print("New transaction received, but this is a wallet node. Ignored.")
        return jsonify({"status": "accepted"}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

# <<< ADDITION >>> 
@app.route("/new_wallet", methods=["POST"])
def receive_new_wallet():
    """
    Let a node broadcast a newly created wallet (public key, region, starting balance).
    We'll store it in our local `wallets` dict, so the validator can find it when processing.
    We do NOT share the private key. This is not how real blockchains track balances,
    but it's a quick fix for the MVP scenario.
    """
    data = request.get_json()
    try:
        pk = data["public_key"]
        region = data["region"]
        balance = float(data["balance"])
        # If we already have this wallet in memory, do nothing:
        if pk in wallets:
            return jsonify({"status": "exists"}), 200

        # Otherwise, create a "dummy" wallet object that references the same public key
        dummy_wallet = Wallet(region)
        dummy_wallet.balance = balance
        # Overwrite the automatically generated public_key with the real one
        dummy_wallet.public_key = ecdsa.VerifyingKey.from_string(bytes.fromhex(pk), curve=ecdsa.SECP256k1)

        # Store it in the local wallets dict
        wallets[pk] = dummy_wallet
        print(f"New wallet received (pk={pk[:16]}... ), region={region}, balance={balance}")
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400

################################################
# Broadcasting
################################################
def broadcast_new_block(block_data: dict):
    """When a validator node confirms a transaction, broadcast the new block to all peers."""
    for peer in peers:
        try:
            url = f"{peer}/new_block"
            response = requests.post(url, json=block_data, timeout=5)
            if response.status_code == 200:
                print(f"Block broadcast to {peer} success.")
            else:
                print(f"Block broadcast to {peer} failed. Status={response.status_code}")
        except Exception as e:
            print(f"Block broadcast to {peer} error: {e}")

def broadcast_new_transaction(tx: Transaction, peer_url=None):
    """If a wallet node wants to submit a transaction to a specific peer (validator)."""
    if peer_url is None:
        for p in peers:
            _submit_transaction_to_peer(tx, p)
    else:
        _submit_transaction_to_peer(tx, peer_url)

def _submit_transaction_to_peer(tx: Transaction, peer_url: str):
    tx_data = {
        "sender": tx.sender,
        "receiver": tx.receiver,
        "amount": tx.amount,
        "fee": tx.fee,
        "data": tx.data,
        "one_validator_confirmation": tx.one_validator_confirmation
    }
    try:
        url = f"{peer_url}/new_transaction"
        response = requests.post(url, json=tx_data, timeout=5)
        if response.status_code == 200:
            print(f"Transaction broadcast to {peer_url} success.")
        else:
            print(f"Transaction broadcast to {peer_url} failed ({response.status_code}).")
    except Exception as e:
        print(f"Could not broadcast to {peer_url}: {e}")

# <<< ADDITION >>> 
def broadcast_new_wallet(pubkey: str, region: str, balance: float):
    """
    After creating a wallet, broadcast the public key, region, and initial balance
    to all peers so that validator nodes can store it in `wallets`.
    """
    wallet_data = {
        "public_key": pubkey,
        "region": region,
        "balance": balance
    }
    for p in peers:
        try:
            url = f"{p}/new_wallet"
            response = requests.post(url, json=wallet_data, timeout=5)
            if response.status_code == 200:
                print(f"Wallet broadcast to {p} success.")
            else:
                print(f"Wallet broadcast to {p} failed ({response.status_code}).")
        except Exception as e:
            print(f"Could not broadcast wallet to {p}: {e}")

################################################
# Validator Node Transaction Processing
################################################
def process_transaction(tx: Transaction) -> bool:
    """
    Full transaction -> block forging routine.
    Only called by validator nodes.
    """
    # 1. Check sender’s wallet & balance
    sender_wallet = wallets.get(tx.sender)
    if not sender_wallet:
        print("Sender wallet not found.")
        return False

    total_deduction = tx.amount + tx.fee
    if sender_wallet.balance < total_deduction:
        print("Insufficient balance.")
        return False

    # Deduct from sender
    sender_wallet.balance -= total_deduction

    # Sender's region => get validator
    sender_validator = get_validator_for_region(sender_wallet.region)
    if not sender_validator:
        print("No validator in sender's region.")
        sender_wallet.balance += total_deduction
        return False
    print(f"Sender's transaction sent to {sender_validator}")

    # Prepare data for signature
    tx_data = json.dumps({
        "sender": tx.sender,
        "receiver": tx.receiver,
        "amount": tx.amount,
        "fee": tx.fee,
        "data": tx.data
    }, sort_keys=True)

    # 2. Sender signs
    # (In reality, the node that created the transaction did the signing. We do an MVP re-sign here for simulation.)
    # But let's do it for consistency:
    tx.sender_signature = sender_wallet.sign(tx_data)
    print("Sender signed the transaction.")

    # 3. Validator verifies
    if not sender_validator.verify_signature(tx_data, tx.sender_signature, tx.sender):
        print("Signature verification failed.")
        release_validator_load(sender_validator)
        sender_wallet.balance += total_deduction
        return False
    print("Signature verified by sender's validator.")

    # 4. Compute tx hash
    tx.transaction_hash = tx.compute_hash()
    print(f"Transaction hash = {tx.transaction_hash}")

    # 5. Broadcast to majority (simulated)
    if not broadcast_to_majority(tx.transaction_hash):
        print("Broadcast to majority failed.")
        release_validator_load(sender_validator)
        sender_wallet.balance += total_deduction
        return False
    print("Broadcast to majority success.")

    # Possibly get receiver’s validator
    receiver_validator = None
    receiver_wallet = wallets.get(tx.receiver)
    if not tx.one_validator_confirmation:
        if not receiver_wallet:
            print("Receiver wallet not found.")
            release_validator_load(sender_validator)
            sender_wallet.balance += total_deduction
            return False
        receiver_validator = get_validator_for_region(receiver_wallet.region)
        if not receiver_validator:
            print("No validator in receiver's region.")
            release_validator_load(sender_validator)
            sender_wallet.balance += total_deduction
            return False
        print(f"Transaction also sent to {receiver_validator}")

        # Check signature again
        if not receiver_validator.verify_signature(tx_data, tx.sender_signature, tx.sender):
            print("Receiver's validator signature check fail.")
            release_validator_load(sender_validator)
            release_validator_load(receiver_validator)
            sender_wallet.balance += total_deduction
            return False
        print("Receiver's validator verified the signature.")

        # Receiver's validator signs
        tx.receiver_validator_signature = receiver_validator.sign(tx.transaction_hash)
        print("Receiver's validator signed the transaction hash.")

    # Sender's validator signs
    tx.sender_validator_signature = sender_validator.sign(tx.transaction_hash)
    print("Sender's validator also signed the transaction hash.")

    # 6. Fee distribution
    if tx.one_validator_confirmation:
        fee_distribution = {f"Validator_{sender_validator.id}": tx.fee}
        sender_validator.balance += tx.fee
    else:
        fee_distribution = {
            f"Validator_{sender_validator.id}": tx.fee/2,
            f"Validator_{receiver_validator.id}": tx.fee/2
        }
        sender_validator.balance += tx.fee/2
        receiver_validator.balance += tx.fee/2
    tx.fee_distribution = fee_distribution
    print(f"Fee distribution: {fee_distribution}")

    # 7. Final consensus
    if not final_consensus(sender_wallet.region, tx.transaction_hash):
        print("Final consensus failed.")
        release_validator_load(sender_validator)
        if receiver_validator:
            release_validator_load(receiver_validator)
        sender_wallet.balance += total_deduction
        return False

    # 8. Mark confirmed, add block
    tx.status = "confirmed"
    block_json = json.loads(tx.serialize())
    blockchain.add_block(block_json)
    print("Transaction confirmed. New block added.")

    # 9. Add the amount to receiver
    if receiver_wallet:
        receiver_wallet.balance += tx.amount

    # 10. Release validator load
    release_validator_load(sender_validator)
    if receiver_validator:
        release_validator_load(receiver_validator)

    # 11. Broadcast block to peers
    broadcast_new_block(block_json)

    return True

def process_mempool_cli():
    """
    Called via CLI to process all transactions in the mempool.
    (Validator node only)
    """
    global mempool
    if not mempool:
        print("Mempool empty.")
        return
    print("Processing mempool...")
    to_remove = []
    for tx in mempool:
        success = process_transaction(tx)
        to_remove.append(tx)  # remove from mempool whether success or fail
    for tx in to_remove:
        if tx in mempool:
            mempool.remove(tx)

################################################
# CLI
################################################
def create_wallet_cli():
    region = random.choice(REGIONS)
    w = Wallet(region)
    pk = w.get_public_key_str()
    wallets[pk] = w
    print(f"Created wallet in {region} with public key={pk}, balance={w.balance} TRIX.")
    # <<< ADDITION >>>
    # Broadcast this new wallet info so peers can store it too.
    broadcast_new_wallet(pk, region, w.balance)
    return pk

def list_wallets_cli():
    if not wallets:
        print("No wallets.")
        return
    for pk, w in wallets.items():
        print(f"PubKey={pk}, region={w.region}, balance={w.balance} TRIX")

def list_blocks_cli():
    blockchain.list_blocks()

def add_peer_cli():
    peer_url = input("Enter peer URL (e.g. http://192.168.1.100:5000): ").strip()
    if peer_url not in peers:
        peers.append(peer_url)
        print(f"Peer {peer_url} added.")
    else:
        print("Peer already in list.")

def list_peers_cli():
    if not peers:
        print("No peers.")
        return
    for p in peers:
        print(p)

def sync_chain_cli():
    global blockchain
    for p in peers:
        try:
            r = requests.get(f"{p}/chain", timeout=5)
            if r.status_code == 200:
                their_chain = r.json()
                if len(their_chain) > len(blockchain.chain):
                    blockchain.chain = their_chain
                    blockchain.save_to_file()
                    print(f"Chain synced from {p}. Local chain length={len(blockchain.chain)}.")
        except Exception as e:
            print(f"Sync from {p} error: {e}")

def send_transaction_cli():
    sender_pk = input("Sender pubkey: ").strip()
    receiver_pk = input("Receiver pubkey: ").strip()
    try:
        amount = float(input("Amount (TRIX): ").strip())
        fee = float(input("Fee (TRIX): ").strip())
    except:
        print("Invalid amount/fee.")
        return
    data = input("Tx data/description: ").strip()
    confirm_mode = input("Confirmation mode (1=one-validator, 2=two-validator): ").strip()
    one_val = True if confirm_mode == "1" else False

    tx = Transaction(sender_pk, receiver_pk, amount, fee, data, one_val)

    if node_mode == "validator":
        mempool.append(tx)
        print("Transaction added to local mempool (validator node).")
    else:
        # If wallet node, broadcast to known validators
        if not peers:
            print("No peers to send to. Add peers first.")
            return
        broadcast_new_transaction(tx)
        print("Transaction broadcast to peers.")

def validator_info_cli():
    """Show local validators (validator node only)."""
    if validators:
        for v in validators:
            print(str(v))
    else:
        print("No local validators.")

def show_help():
    print("\nCommands:")
    if node_mode == "validator":
        print("  create_wallet       - Create a new wallet locally (also broadcasts it)")
        print("  send_transaction    - Create a transaction and add to local mempool")
        print("  process_mempool     - Process all transactions in mempool (forge blocks)")
        print("  validator_info      - List local validators & balances")
    else:
        print("  create_wallet       - Create a new wallet locally (also broadcasts it)")
        print("  send_transaction    - Create a transaction and broadcast to validator peers")
    print("  list_wallets        - Show local wallets/balances")
    print("  list_blocks         - Show local chain")
    print("  add_peer            - Add a peer node URL")
    print("  list_peers          - List known peer URLs")
    print("  sync_chain          - Sync chain from peers")
    print("  exit                - Quit\n")

def cli_loop():
    show_help()
    while True:
        cmd = input("Enter command: ").strip().lower()
        if cmd == "exit":
            print("Exiting CLI.")
            break
        elif cmd == "create_wallet":
            create_wallet_cli()
        elif cmd == "list_wallets":
            list_wallets_cli()
        elif cmd == "list_blocks":
            list_blocks_cli()
        elif cmd == "add_peer":
            add_peer_cli()
        elif cmd == "list_peers":
            list_peers_cli()
        elif cmd == "sync_chain":
            sync_chain_cli()
        elif cmd == "send_transaction":
            send_transaction_cli()
        elif cmd == "process_mempool":
            if node_mode == "validator":
                process_mempool_cli()
            else:
                print("This node is not a validator. No mempool to process.")
        elif cmd == "validator_info":
            if node_mode == "validator":
                validator_info_cli()
            else:
                print("This node is a wallet node, no local validators.")
        else:
            print("Unknown command.")
            show_help()

################################################
# Main
################################################
def run_server():
    """Start Flask app for P2P endpoint handling."""
    app.run(host="0.0.0.0", port=5000)

def parse_args():
    """
    python trix_blockchain.py --mode wallet  --server
    python trix_blockchain.py --mode validator --server
    """
    global node_mode
    if "--mode" in sys.argv:
        i = sys.argv.index("--mode")
        if i+1 < len(sys.argv):
            node_mode_candidate = sys.argv[i+1]
            if node_mode_candidate in ["wallet", "validator"]:
                node_mode = node_mode_candidate
    if node_mode == "validator":
        init_validators()
        print("Node mode: VALIDATOR => local validators & mempool active.")
    else:
        print("Node mode: WALLET => no local validators, no mempool.")

def main():
    parse_args()
    # If run with "--server" argument, start the Flask server in a background thread
    if "--server" in sys.argv:
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print("Flask server started on port 5000.")

    # Start the CLI
    cli_loop()

if __name__ == '__main__':
    main()
