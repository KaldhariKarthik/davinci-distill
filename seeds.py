"""
SEED REQUESTS — the one file you own.

The teacher expands each seed into many training examples (VARIATIONS_PER_SEED),
filling the {slots}. EDIT freely — cut, add, reword.

Each seed:
  - "request":  what the user asks (with {slots} the generator fills)
  - "source":   'files' | 'web' | 'calendar' | 'none'  (what context to pair)
  - "skill":    the reasoning-over-context sub-skill being taught
"""

SEEDS = [
    # ---- personal context: grounded answer from files -----------------------
    {"request": "What did I decide about {topic} in my notes?",
     "source": "files", "skill": "grounded_answer"},
    {"request": "Did I follow up with {person} about {thing}?",
     "source": "files", "skill": "grounded_answer"},
    {"request": "What were the action items from my last meeting about {project}?",
     "source": "files", "skill": "grounded_answer"},
    {"request": "Remind me what {person} said about {thing}.",
     "source": "files", "skill": "grounded_answer"},
    {"request": "What's the status of {project} based on my notes?",
     "source": "files", "skill": "synthesis"},

    # ---- personal context: summarize ----------------------------------------
    {"request": "Find what I wrote about {project} and summarize it.",
     "source": "files", "skill": "summarize"},
    {"request": "Give me a quick summary of my notes on {topic}.",
     "source": "files", "skill": "summarize"},
    {"request": "Summarize the key points from the {thing} document.",
     "source": "files", "skill": "summarize"},

    # ---- calendar / planning ------------------------------------------------
    {"request": "What's on my calendar today and what should I prioritize?",
     "source": "calendar", "skill": "synthesis_planning"},
    {"request": "I have {deadline} on Friday - plan my week around it.",
     "source": "calendar", "skill": "synthesis_planning"},
    {"request": "When am I free this week for a {duration} meeting?",
     "source": "calendar", "skill": "grounded_answer"},
    {"request": "What's my busiest day this week?",
     "source": "calendar", "skill": "grounded_answer"},
    {"request": "Given my schedule, when should I work on {project}?",
     "source": "calendar", "skill": "synthesis_planning"},

    # ---- web: plan / synthesize ---------------------------------------------
    {"request": "Plan a weekend trip to {place}.",
     "source": "web", "skill": "synthesis_planning"},
    {"request": "Plan a {duration} itinerary for {place}.",
     "source": "web", "skill": "synthesis_planning"},
    {"request": "Summarize what the latest news says about {topic}.",
     "source": "web", "skill": "synthesis"},
    {"request": "What are the main things to know about {topic}?",
     "source": "web", "skill": "synthesis"},

    # ---- web: compare / recommend (junk filtering) --------------------------
    {"request": "Compare {option_a} and {option_b} for me.",
     "source": "web", "skill": "junk_filter_recommend"},
    {"request": "Given these results, recommend the best {thing}.",
     "source": "web", "skill": "junk_filter_recommend"},
    {"request": "Which is better for {use_case}, {option_a} or {option_b}?",
     "source": "web", "skill": "junk_filter_recommend"},
    {"request": "What's the best {thing} for {use_case} on a budget?",
     "source": "web", "skill": "junk_filter_recommend"},

    # ---- web: guidance / how-to ---------------------------------------------
    {"request": "How do I make {dish}?",
     "source": "web", "skill": "guidance_steps"},
    {"request": "Walk me through {task} step by step.",
     "source": "web", "skill": "guidance_steps"},
    {"request": "Help me organize {space}.",
     "source": "web", "skill": "guidance_steps"},
    {"request": "What's the best way to {activity}?",
     "source": "web", "skill": "guidance_steps"},
    {"request": "Give me a beginner's guide to {activity}.",
     "source": "web", "skill": "guidance_steps"},

    # ---- reasoning over mixed / contradictory context -----------------------
    {"request": "These sources disagree about {topic} - what's the real answer?",
     "source": "web", "skill": "junk_filter_recommend"},
    {"request": "Based on everything, what should I do about {thing}?",
     "source": "files", "skill": "synthesis"},

    # ---- tool routing: intent + parameter extraction ------------------------
    {"request": "Add a reminder to {action} tomorrow at {time}.",
     "source": "none", "skill": "intent_extract"},
    {"request": "Schedule {event} for {day} at {time}.",
     "source": "none", "skill": "intent_extract"},
    {"request": "Draft an email to {person} about {topic}.",
     "source": "files", "skill": "intent_extract_grounded"},
    {"request": "Play some {music_type} music.",
     "source": "none", "skill": "intent_extract"},
    {"request": "Set a timer for {duration}.",
     "source": "none", "skill": "intent_extract"},
    {"request": "Add {thing} to my task list.",
     "source": "none", "skill": "intent_extract"},
]

SLOT_FILLERS = {
    "topic":     ["the budget", "the Q3 plan", "the vendor choice", "the launch date",
                  "the hiring plan", "the pricing", "the marketing strategy"],
    "project":   ["the DaVinci prototype", "the grant proposal", "the client work",
                  "the app redesign", "the research paper"],
    "person":    ["Priya", "the landlord", "my manager", "the vendor", "Rahul",
                  "the client", "my co-founder"],
    "thing":     ["the contract", "the invoice", "the schedule", "the design",
                  "the proposal", "the report", "the agreement"],
    "place":     ["Goa", "Manali", "Jaipur", "Kerala", "Rishikesh", "Udaipur", "Ladakh"],
    "dish":      ["pizza", "biryani", "pasta", "dosa", "butter chicken", "ramen"],
    "option_a":  ["the A100", "iPhone", "Plan A", "React", "renting", "solar"],
    "option_b":  ["the H100", "Android", "Plan B", "Vue", "buying", "grid power"],
    "use_case":  ["gaming", "ML training", "a small team", "daily commuting",
                  "a startup", "home use"],
    "deadline":  ["a grant submission", "a product demo", "a report", "a client pitch"],
    "duration":  ["30-minute", "one hour", "two day", "three day", "weekend"],
    "task":      ["setting up a tent", "changing a tyre", "fixing a leak",
                  "assembling furniture", "starting a garden"],
    "space":     ["my desk", "my kitchen", "my closet", "my garage", "my inbox"],
    "activity":  ["learning guitar", "meal prepping", "running a 5k",
                  "budgeting", "waking up early"],
    "action":    ["call the landlord", "submit the form", "pay the bill",
                  "email the client", "book the tickets"],
    "event":     ["a dentist appointment", "a team sync", "a call with Priya",
                  "a demo", "lunch"],
    "day":       ["Monday", "Tuesday", "Thursday", "next Friday"],
    "time":      ["10 AM", "2 PM", "9 in the morning", "4:30", "6 PM"],
    "music_type":["jazz", "lo-fi", "classical", "rock", "focus"],
}
