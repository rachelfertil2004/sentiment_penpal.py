
from transformers import pipeline
import random
from typing import Tuple

# Load sentiment model (Twitter RoBERTa)
analyzer = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)

# Tone templates
TEMPLATES = {
    "supportive": {
        "positive": [
            "Love that for you — congrats! 🌟 Anything you want to build on next?",
            "So good to hear. Savor it — you earned this! 🙌"
        ],
        "neutral": [
            "Got it. If you want to unpack this more, I’m here.",
            "Thanks for sharing. What outcome would feel like a win?"
        ],
        "negative": [
            "That sounds tough. One step at a time — what’s the smallest next move?",
            "I’m sorry you’re dealing with that. Want to talk options together?"
        ],
    },
    "upbeat": {
        "positive": [
            "Vibes are immaculate ✨ Keep the momentum rolling!",
            "Let’s gooo! 🔥 What’s the next milestone?"
        ],
        "neutral": [
            "Noted! Let’s turn this into a mini-win — what’s one quick action?",
            "I’m game. How do we make this 1 % better today?"
        ],
        "negative": [
            "We’ll bounce back. Reset, refocus, rally. You’ve got this 💪",
            "Tough beat — but the plot-twist comeback is loading…"
        ],
    },
    "witty": {
        "positive": [
            "Certified good-news alert 🚨 Proceed to happy dance protocol.",
            "Your serotonin called; it says ‘keep doing that.’"
        ],
        "neutral": [
            "I read you. Let’s sprinkle strategy on this and stir.",
            "Acknowledged. Consider this the loading screen before progress."
        ],
        "negative": [
            "Oof.exe — sending patches and snacks.",
            "When life throws lemons, we A/B test lemonade recipes."
        ],
    }
}

def get_sentiment(text: str) -> Tuple[str, float]:
    preds = analyzer(text)
    if isinstance(preds[0], list):
        preds = preds[0]
    top = max(preds, key=lambda x: x['score'])
    label = top['label'].lower()
    score = float(top['score'])
    return label, score

def penpal_reply(text: str, tone: str = "supportive") -> dict:
    if not text.strip():
        return {"error": "Please enter some text."}
    sentiment, score = get_sentiment(text)
    reply = random.choice(TEMPLATES[tone][sentiment])
    return {
        "input": text,
        "sentiment": sentiment,
        "confidence": round(score, 4),
        "tone": tone,
        "reply": reply
    }

if __name__ == "__main__":
    # Simple demo
    sample_text = "I didn’t get the role I wanted. Feeling a bit discouraged."
    out = penpal_reply(sample_text, tone="supportive")
    print("Input:", out["input"])
    print("Detected:", out["sentiment"], f"({out['confidence']*100:.1f}% )")
    print("Tone:", out["tone"])
    print("Reply:", out["reply"])
