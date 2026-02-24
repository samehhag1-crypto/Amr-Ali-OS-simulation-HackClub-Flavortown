import random, copy, time
from dataclasses import dataclass
from typing import List

@dataclass
class WorldEvent:
    name: str
    effect: float
    turns: int = 1

class EventManager:
    def __init__(self):
        self.active = []
        self.pool = [
            WorldEvent("Market spike", 0.15, 2),
            WorldEvent("System meltdown", -0.2, 2),
            WorldEvent("Minor crash", -0.1, 1),
            WorldEvent("Big discovery", 0.25, 1),
            WorldEvent("Energy surge", 0.1, 3)
        ]

    def maybe_trigger(self):
        if random.random() < 0.18:
            e = copy.deepcopy(random.choice(self.pool))
            self.active.append(e)
            print(f"\n🔥 Event popped: {e.name}")
            return e
        return None

    def update(self):
        for e in self.active: e.turns -= 1
        self.active = [e for e in self.active if e.turns > 0]

def boost_agent(agent, events: EventManager):
    if random.random() < 0.3: agent.energy += 0.25
    for e in events.active: agent.energy += e.effect
    agent.energy = max(0, min(agent.energy, 20))
    if random.random() < 0.12:
        agent.move(random.choice([-1,0,1]), random.choice([-1,0,1]))

@dataclass
class Mission:
    text: str
    reward: float
    done: bool = False

class MissionManager:
    def __init__(self):
        self.list: List[Mission] = []

    def add_random(self):
        options = [
            ("Push wealth past 1500", 0.1),
            ("Keep all agents alive", 0.07),
            ("Get max curiosity", 0.08),
            ("Boost mental strength", 0.09),
            ("Make agents explore", 0.05)
        ]
        t,r = random.choice(options)
        m = Mission(t,r)
        self.list.append(m)
        print(f"🚀 Mission added: {t}")
        return m

    def check(self, os_ref):
        for m in self.list:
            if m.done: continue
            txt = m.text.lower()
            if "wealth" in txt and os_ref.wealth.capital > 1500: self.complete(m, os_ref)
            elif "alive" in txt and all(a.energy>0 for a in os_ref.world.agents): self.complete(m, os_ref)
            elif "curiosity" in txt and os_ref.emotions.state["curiosity"]>0.8: self.complete(m, os_ref)
            elif "mental" in txt and os_ref.growth.metrics["mental_strength"]>0.7: self.complete(m, os_ref)
            elif "explore" in txt and any(a.energy<19 for a in os_ref.world.agents): self.complete(m, os_ref)

    def complete(self, m, os_ref):
        m.done = True
        os_ref.growth.metrics["knowledge"] = min(1.0, os_ref.growth.metrics["knowledge"] + m.reward)
        os_ref.growth.metrics["wealth"] = min(1.0, os_ref.growth.metrics["wealth"] + m.reward/2)
        os_ref.logger.log(f"🎉 Mission done: {m.text}")

def extra_stats(os_ref):
    agents = os_ref.world.agents
    print("\n--- Stats check ---")
    print("Agents alive:", len(agents))
    if agents:
        e = [a.energy for a in agents]
        print("Avg energy:", round(sum(e)/len(e),2))
        print("Max energy:", round(max(e),2))
        print("Min energy:", round(min(e),2))
    active = [m.text for m in os_ref.missions.list if not m.done]
    done = [m.text for m in os_ref.missions.list if m.done]
    print("Active missions:", active)
    print("Completed missions:", done)
    print("------------------\n")

def step_all(os_ref, events: EventManager):
    events.maybe_trigger()
    for a in os_ref.world.agents: boost_agent(a, events)
    events.update()

def random_vm_script():
    ops = ["push","add","sub","mul","div","print"]
    return "\n".join(random.choice(ops) + (f" {random.randint(1,50)}" if ops=="push" else "") for _ in range(random.randint(3,7)))

def slight_evolution(os_ref):
    for a in os_ref.world.agents:
        if random.random() < 0.05:
            a.energy += 1
            for layer in a.brain.layers:
                for i in range(len(layer)):
                    for j in range(len(layer[i])):
                        if random.random()<0.04: layer[i][j] += random.uniform(-0.08,0.08)
        a.energy = min(a.energy,20)

def integrate(os_ref):
    os_ref.events = EventManager()
    os_ref.missions = MissionManager()
    for _ in range(3): os_ref.missions.add_random()