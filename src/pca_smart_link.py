"""
This FastAPI application implements two sampling strategies for a multi-armed bandit problem:
1. Epsilon-Greedy Algorithm
2. Upper Confidence Bound (UCB) Algorithm

It allows for experimenting with different strategies to optimize the selection of offers based on click data.
The application includes endpoints for sampling offers, recording feedback, and retrieving offer statistics.
"""

import math
import random
import uvicorn
from fastapi import FastAPI

app = FastAPI()
epsilon = 0.1  # Exploration factor for Epsilon-Greedy Algorithm
click_offer_dic = {}  # Maps click IDs to offer IDs
offer_clicks = {}  # Tracks the number of clicks per offer
offer_rewards = {}  # Tracks the total rewards per offer
offer_conversion = {}  # Tracks the number of conversions per offer


@app.on_event("startup")
async def startup_event():
    """Clears all the tracking dictionaries on application startup."""
    global click_offer_dic, offer_clicks, offer_rewards, offer_conversion
    click_offer_dic.clear()
    offer_clicks.clear()
    offer_rewards.clear()
    offer_conversion.clear()


@app.get("/sample_epsilon_greedy/")
async def sample_epsilon_greedy(click_id: int, offer_ids: str) -> dict:
    """
    Endpoint for Epsilon-Greedy sampling strategy. It randomly explores new offers with a probability of epsilon,
    and exploits the best-known offer otherwise.
    """
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    # Epsilon-Greedy strategy
    if random.random() < epsilon:
        offer_id = random.choice(offers_ids)
        sampler = "random"
    else:
        rpc_values = {offer: (offer_rewards.get(offer, 0) / max(1, offer_clicks.get(offer, 1)))
                      for offer in offers_ids}
        offer_id = max(rpc_values, key=rpc_values.get, default=offers_ids[0])
        sampler = "greedy"

    # Update tracking dictionaries
    click_offer_dic[click_id] = offer_id
    offer_clicks[offer_id] = offer_clicks.setdefault(offer_id, 0) + 1

    # Prepare response
    return {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": sampler,
    }


@app.get("/sample_upper_confidence_bound/")
async def sample_upper_confidence_bound(click_id: int, offer_ids: str) -> dict:
    """
    Endpoint for Upper Confidence Bound (UCB) sampling strategy. It balances exploration and exploitation
    by considering both the average reward and the uncertainty of each offer.
    """
    offers_ids = [int(offer) for offer in offer_ids.split(",")]

    total_clicks = sum(offer_clicks.values())
    # UCB calculation
    rpc_values = {offer: (offer_rewards.get(offer, 0) / max(1, offer_clicks.get(offer, 1))) + math.sqrt(
        2 * math.log(total_clicks + 1) / max(1, offer_clicks.get(offer, 1)))
                  for offer in offers_ids}
    offer_id = max(rpc_values, key=rpc_values.get, default=offers_ids[0])
    sampler = "ucb"

    # Update tracking dictionaries
    click_offer_dic[click_id] = offer_id
    offer_clicks[offer_id] = offer_clicks.setdefault(offer_id, 0) + 1

    # Prepare response
    return {
        "click_id": click_id,
        "offer_id": offer_id,
        "sampler": sampler,
    }


@app.put("/feedback/")
async def feedback(click_id: int, reward: float) -> dict:
    """
    Endpoint to record feedback for a click. It updates the rewards and conversion data based on the received feedback.
    """
    offer_id = None
    is_conversion = False
    if click_id in click_offer_dic:
        offer_id = click_offer_dic.pop(click_id)
        is_conversion = reward > 0  # Determine if there was a conversion
        if is_conversion:
            # Update conversion and reward data
            offer_conversion[offer_id] = offer_conversion.setdefault(offer_id, 0) + 1
            offer_rewards[offer_id] = offer_rewards.setdefault(offer_id, 0) + reward

    # Prepare response
    return {
        "click_id": click_id,
        "offer_id": offer_id,
        "is_conversion": is_conversion,
        "reward": float(reward)
    }


@app.get("/offer_ids/{offer_id}/stats/")
async def stats(offer_id: int) -> dict:
    """
    Endpoint to retrieve statistics for a specific offer. It returns data like clicks, conversions, total reward,
    conversion rate (CR), and revenue per click (RPC).
    """
    clicks = offer_clicks.get(offer_id, 0)
    conversions = offer_conversion.get(offer_id, 0)
    reward = offer_rewards.get(offer_id, 0)

    # Calculate statistics
    return {
        "offer_id": offer_id,
        "clicks": clicks,
        "conversions": conversions,
        "reward": float(reward),
        "cr": float(max(0, conversions) / max(1, clicks)),
        "rpc": float(max(0, reward) / max(1, clicks)),
    }


def main() -> None:
    """Runs the FastAPI application."""
    uvicorn.run("pca_smart_link:app", host="localhost")


if __name__ == "__main__":
    main()
