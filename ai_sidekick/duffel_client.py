import os
from duffel_api import Duffel

DUFFEL_API_KEY = os.getenv("DUFFEL_API_KEY")

duffel = Duffel(access_token= DUFFEL_API_KEY)

OFFER_CONTEXT = {}

def search_flights(origin: str, destination: str, departure_date: str, passengers: int = 1):
    try: 
        offers_req = duffel.offer_requests.create(
            slices = [
                {"origin": origin, "destination": destination, "departure_date": departure_date} 
            ],
            passengers = [{"type": "adult"} for _ in range (passengers)],
            cabin_class = "Economy"
        )
        
        offers = []
        for offer in offers_req.offers[:5] :
            OFFER_CONTEXT[offer.id] = {
                "passenger_ids": [p.id for p in offers_req.passengers] 
            }

        for sl in offer.slices:
            print(f"for offer in slices{offer.id}: {sl.segments} ")    
            offers.append({
                "id": offer.id,
                "Airline": offer.owner.name,
                "total_ammount": f"{offer.total_amount}, {offer.total_currency}",
                "slices":[
                    {
                        "origin": sl.origin.iata_code,
                        "destination": sl.destination.iata_code,
                        "departure": sl.segments[0].departing_at,
                        "arrival": sl.segments[-1].arriving_at,
                        "duration": sl.duration
                    } for sl in offer.slices
                    
                ]
            })

            
        
        return offers if offers else "No flights found"
    except Exception as e:
        return f"Flight search error: {str(e)}"


def book_flight(offer_id: str, passengers_info: list, contact: dict, payment: dict | None = None):
    """
    Create an order for a previously returned offer_id.
    passengers_info: list of dicts with at least given_name, family_name, born_on (YYYY-MM-DD)
    contact: dict with email, phone_number, given_name, family_name
    payment (optional): {"type":"balance","amount":"123.45","currency":"USD"}  (or omit in test if you don't have balance)
    """
    ctx = OFFER_CONTEXT.get(offer_id)
    if not ctx:
        return {"error": "Unknown offer_id. Please search again and select from the latest offers."}

    pax_ids = ctx["passenger_ids"]
    # Ensure we have the same count as initial search
    if len(passengers_info) != len(pax_ids):
        # pad or truncate to match
        passengers_info = (passengers_info + [{}] * len(pax_ids))[:len(pax_ids)]

    passengers = []
    for pax_id, info in zip(pax_ids, passengers_info):
        passengers.append({
            "id": pax_id,
            "title": info.get("title", "mr"),
            "given_name": info.get("given_name", "Test"),
            "family_name": info.get("family_name", "Passenger"),
            "born_on": info.get("born_on", "1990-01-01"),
            "phone_number": info.get("phone_number", contact.get("phone_number")),
            "email": info.get("email", contact.get("email")),
        })

    payload = {
        "selected_offers": [offer_id],
        "passengers": passengers,
        "contact": {
            "email": contact.get("email"),
            "phone_number": contact.get("phone_number"),
            "given_name": contact.get("given_name", passengers[0]["given_name"]),
            "family_name": contact.get("family_name", passengers[0]["family_name"]),
        }
    }
    if payment:
        payload["payments"] = [payment]

    # Remove empty keys
    payload = {k: v for k, v in payload.items() if v}

    try:
        order = duffel.orders.create(**payload)
        summary = {
            "order_id": order.id,
            "booking_reference": getattr(order, "booking_reference", None),
            "airline": getattr(order.owner, "name", None),
            "total_amount": order.total_amount,
            "currency": order.total_currency,
            "status": getattr(order, "status", None),
            "passengers": [{"name": f"{p.given_name} {p.family_name}", "born_on": p.born_on} for p in order.passengers],
            "slices": [
                {
                    "origin": sl.origin.iata_code,
                    "destination": sl.destination.iata_code,
                    "departure": sl.segments[0].departing_at,
                    "arrival": sl.segments[-1].arriving_at,
                } for sl in order.slices
            ],
        }
        return summary
    except Exception as e:
        return {"error": f"Booking failed: {e}"}    
