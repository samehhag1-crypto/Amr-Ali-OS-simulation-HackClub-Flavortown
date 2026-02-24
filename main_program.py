import sys
import os
import math
import time
import random
import json
import sqlite3
import socket
import threading
import curses
import uuid
import copy
from dataclasses import dataclass
from typing import List, Dict

# import enhancements
from enhancements import EventManager, boost_agent, MissionManager, extra_stats, step_all, random_vm_script, slight_evolution, integrate

CREATOR = "Amr Ali"
VERSION = "AMR-OS 4.0 Teen Edition"

DNA = {
    "discipline": 0.87,
    "ambition": 0.95,
    "curiosity": 0.98,
    "resilience": 0.90,
    "focus": 0.82,
    "risk_tolerance": 0.75
}

DB_FILE = "amr_os_teen.db"
NODE_PORT = 6001

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    return y * (1 - y)

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def random_vector(n):
    return [random.uniform(-1, 1) for _ in range(n)]

class MemoryEngine:
    def __init__(self):
        self.conn = sqlite3.connect(DB_FILE)
        self._init_db()
                            
    def _init_db(self):
        c = self.conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS memory(
            id TEXT PRIMARY KEY,
            content TEXT,
            reward REAL,
            timestamp REAL
        )
        """)
        self.conn.commit()
                
    def store(self, content, reward=0.0):
        try:
            c = self.conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO memory VALUES (?, ?, ?, ?)",
                (str(uuid.uuid4()), content, reward, time.time())
            )
            self.conn.commit()
        except Exception:
            pass
                
    def search(self, keyword):
        c = self.conn.cursor()
        c.execute(
            "SELECT content FROM memory WHERE content LIKE ?",
            ('%' + keyword + '%',)
        )
        return [row[0] for row in c.fetchall()]

class KnowledgeGraph:
    def __init__(self):
        self.nodes: Dict[str, set] = {}
                            
    def link(self, a, b):
        self.nodes.setdefault(a, set()).add(b)
        self.nodes.setdefault(b, set())
                
    def query(self, term):
        return [n for n in self.nodes if term in n]

class NeuralNet:
    def __init__(self, sizes):
        self.sizes = sizes
        self.layers = []
        for i in range(len(sizes)-1):
            self.layers.append([
                [random.uniform(-1,1) for _ in range(sizes[i+1])]
                for _ in range(sizes[i])
            ])
        self.lr = 0.01

    def forward(self, x):
        self.activations = [x]
        for layer in self.layers:
            next_x = []
            for j in range(len(layer[0])):
                s = sum(x[i] * layer[i][j] for i in range(len(x)))
                next_x.append(sigmoid(s))
            x = next_x
            self.activations.append(x)
        return x

    def backward(self, target):
        deltas = []

        output = self.activations[-1]
        delta = [(output[i] - target[i]) * dsigmoid(output[i])
                 for i in range(len(output))]
        deltas.append(delta)

        for l in reversed(range(len(self.layers)-1)):
            layer = self.layers[l]
            next_layer = self.layers[l+1]
            delta = []
            for i in range(len(layer)):
                error = sum(
                    deltas[-1][j] * next_layer[i][j]
                    for j in range(len(next_layer[0]))
                )
                delta.append(error * dsigmoid(self.activations[l+1][i]))
            deltas.append(delta)

        deltas.reverse()

        for l in range(len(self.layers)):
            for i in range(len(self.layers[l])):
                for j in range(len(self.layers[l][0])):
                    self.layers[l][i][j] -= (
                        self.lr *
                        deltas[l][j] *
                        self.activations[l][i]
                    )

class EmotionSystem:
    def __init__(self):
        self.state = {
            "confidence": DNA["discipline"],
            "drive": DNA["ambition"],
            "curiosity": DNA["curiosity"],
            "fear": 0.2
        }
                            
    def update(self, reward):
        self.state["confidence"] = clamp(self.state["confidence"] + reward*0.05)
        self.state["drive"] = clamp(self.state["drive"] + reward*0.07)
        self.state["fear"] = clamp(self.state["fear"] - reward*0.03)
        self.state["curiosity"] = clamp(self.state["curiosity"] + random.uniform(-0.02,0.02))

class GrowthSystem:
    def __init__(self):
        self.metrics = {
            "knowledge": 0.2,
            "wealth": 0.1,
            "strategy": 0.3,
            "discipline": DNA["discipline"],
            "mental_strength": 0.5
        }
                            
    def reinforce(self, reward):
        if reward>0:
            self.metrics["knowledge"]=clamp(self.metrics["knowledge"]+0.01)
            self.metrics["wealth"]=clamp(self.metrics["wealth"]+0.005)
        else:
            self.metrics["mental_strength"]=clamp(self.metrics["mental_strength"]+0.01)

@dataclass
class Goal:
    text: str
    priority: float
    progress: float = 0.0

class GoalSystem:
    def __init__(self):
        self.goals: List[Goal] = []
                            
    def add(self, text, priority=1.0):
        self.goals.append(Goal(text, priority))
                
    def next_goal(self):
        if not self.goals:
            return None
        return sorted(self.goals, key=lambda g: g.priority*(1-g.progress), reverse=True)[0]

class WealthEngine:
    def __init__(self, growth_system: GrowthSystem, emotions: EmotionSystem):
        self.capital = 1000.0
        self.growth = growth_system
        self.emotions = emotions
        self.history = []
                            
    def simulate_trade(self):
        risk_factor = DNA["risk_tolerance"] * self.emotions.state["confidence"]
        trend = (self.strategy_score() / 1000.0)
        volatility = (random.uniform(-0.5,0.5) + trend) * 0.1
        change = self.capital * volatility * risk_factor
        self.capital += change
        reward = change / 1000.0
        self.growth.reinforce(reward)
        self.history.append(self.capital)
        return change

    def strategy_score(self):
        if len(self.history)<2: return 0
        return self.history[-1]-self.history[0]

class StrategyEngine:
    def __init__(self, net: NeuralNet):
        self.net = net
                            
    def evaluate(self, features: List[float]) -> float:
        out = self.net.forward(features)
        score = sum(out)/len(out)
        return score

class World:
    def __init__(self, size=20):
        self.size = size
        self.grid = [[None for _ in range(size)] for _ in range(size)]
        self.agents: List[Agent] = []
                            
    def add_agent(self, agent):
        self.agents.append(agent)
        self.grid[agent.x][agent.y] = agent
                
    def step(self):
        for agent in list(self.agents):
            agent.step(self)
            if agent.energy <=0:
                self.grid[agent.x][agent.y]=None
                self.agents.remove(agent)
                                                            
    def random_empty_cell(self):
        empty = [
            (i, j)
            for i in range(self.size)
            for j in range(self.size)
            if self.grid[i][j] is None
        ]
        return random.choice(empty) if empty else None
                                        
class Agent:
    def __init__(self, world: World):
        self.world = world
        cell = world.random_empty_cell()
        if cell:
            self.x, self.y = cell
        else:
            self.x, self.y = 0, 0
        self.energy=random.uniform(5,15)
        self.brain=NeuralNet([4,6,2])
                            
    def sense(self):
        return [self.x/self.world.size,
                self.y/self.world.size,
                self.energy/20,
                random.random()]

    def move(self, dx, dy):
        nx=clamp(self.x+dx,0,self.world.size-1)
        ny=clamp(self.y+dy,0,self.world.size-1)
        if self.world.grid[int(nx)][int(ny)] is None:
            self.world.grid[self.x][self.y]=None
            self.x,self.y=int(nx),int(ny)
            self.world.grid[self.x][self.y]=self
                                        
    def step(self, world):
        inputs=self.sense()
        out=self.brain.forward(inputs)
        dx=1 if out[0]>0.5 else -1
        dy=1 if out[1]>0.5 else -1
        self.move(dx,dy)
        self.energy-=0.2
        reward = 1.0 if self.energy > 0 else 0.0
        self.brain.backward([reward, reward])
                
class WorldEvolution:
    def __init__(self, world: World):
        self.world = world
                            
    def evolve(self):
        if len(self.world.agents)<2: return
        sorted_agents=sorted(self.world.agents,key=lambda a:a.energy,reverse=True)
        top=sorted_agents[:len(sorted_agents)//2]
        bottom=sorted_agents[len(sorted_agents)//2:]
        for agent in bottom:
            self.world.grid[agent.x][agent.y]=None
            if agent in self.world.agents: self.world.agents.remove(agent)
        for agent in top:
            if len(self.world.agents)<30:
                clone=Agent(self.world)
                clone.brain = copy.deepcopy(agent.brain)
                for layer in clone.brain.layers:
                    for i in range(len(layer)):
                        for j in range(len(layer[0])):
                            if random.random() < 0.05:
                                layer[i][j] += random.uniform(-0.2, 0.2)
                self.world.add_agent(clone)
                                                                                    
class TrainingLoop:
    def __init__(self, net: NeuralNet, emotions: EmotionSystem,
                growth: GrowthSystem, wealth: WealthEngine, world: World):
        self.net=net
        self.emotions=emotions
        self.growth=growth
        self.wealth=wealth
        self.world=world
                            
    def step(self):
        x = [
            self.wealth.capital / 5000,
            self.emotions.state["confidence"],
            self.growth.metrics["knowledge"],
            self.growth.metrics["mental_strength"],
            random.random()
        ]

        y = [
            1.0 if self.wealth.capital > 1000 else 0.0,
            self.emotions.state["confidence"]
        ]

        self.net.forward(x)
        self.net.backward(y)
        reward=random.uniform(-1,1)
        self.emotions.update(reward)
        self.growth.reinforce(reward)
        self.wealth.simulate_trade()
        self.world.step()
                
class Logger:
    def __init__(self):
        self.logs=[]
                            
    def log(self,msg):
        ts=time.strftime("%H:%M:%S")
        s=f"[{ts}] {msg}"
        self.logs.append(s)
        print(s)
                
    def last(self,n=5):
        return self.logs[-n:]

class StackVM:
    def __init__(self):
        self.stack=[]
        self.vars={}
        self.pc=0
                            
    def run(self, bytecode):
        self.pc=0
        while self.pc<len(bytecode):
            instr=bytecode[self.pc]
            op=instr[0]
            try:
                if op=="PUSH":
                    self.stack.append(instr[1])
                elif op=="ADD":
                    b=self.stack.pop()
                    a=self.stack.pop()
                    self.stack.append(a+b)
                elif op=="SUB":
                    b=self.stack.pop()
                    a=self.stack.pop()
                    self.stack.append(a-b)
                elif op=="MUL":
                    b=self.stack.pop()
                    a=self.stack.pop()
                    self.stack.append(a*b)
                elif op=="DIV":
                    b=self.stack.pop()
                    a=self.stack.pop()
                    self.stack.append(a/b if b!=0 else 0)
                elif op=="STORE":
                    val=self.stack.pop()
                    self.vars[instr[1]]=val
                elif op=="LOAD":
                    self.stack.append(self.vars.get(instr[1],0))
                elif op=="PRINT":
                    print(self.stack.pop())
                elif op=="JUMP":
                    self.pc=instr[1]
                    continue
                elif op=="JZ":
                    val=self.stack.pop()
                    if val==0:
                        self.pc=instr[1]
                        continue
            except:
                pass
            self.pc+=1
                
class ScriptCompiler:
    def compile(self, code:str):
        lines=code.split("\n")
        bytecode=[]
        for line in lines:
            parts=line.strip().split()
            if not parts: continue
            cmd=parts[0].lower()
            if cmd=="push": bytecode.append(("PUSH",float(parts[1])))
            elif cmd=="add": bytecode.append(("ADD",))
            elif cmd=="sub": bytecode.append(("SUB",))
            elif cmd=="mul": bytecode.append(("MUL",))
            elif cmd=="div": bytecode.append(("DIV",))
            elif cmd=="store": bytecode.append(("STORE",parts[1]))
            elif cmd=="load": bytecode.append(("LOAD",parts[1]))
            elif cmd=="print": bytecode.append(("PRINT",))
        return bytecode
                            
class EmotionNode(threading.Thread):
    def __init__(self, emotions: EmotionSystem):
        super().__init__(daemon=True)
        self.emotions=emotions
                            
    def run(self):
        s=socket.socket()
        try:
            s.bind(("localhost",NODE_PORT))
            s.listen(5)
            while True:
                conn, _ = s.accept()
                data=json.dumps(self.emotions.state)
                try: conn.send(data.encode())
                except: pass
                conn.close()
        except:
            pass
                                
class Dashboard:
    def __init__(self, os_ref):
        self.os=os_ref
                            
    def start(self):
        curses.wrapper(self.loop)
                
    def loop(self,std):
        try:
            while True:
                std.clear()
                std.addstr(0,0,f"{VERSION}")
                std.addstr(1,0,f"Creator: {CREATOR}")
                std.addstr(2,0,f"Wealth: {round(self.os.wealth.capital,2)}")
                std.addstr(3,0,f"Goals: {len(self.os.goals.goals)}")
                std.addstr(4,0,f"Emotion: {self.os.emotions.state}")
                std.addstr(5,0,f"Growth: {self.os.growth.metrics}")
                offset=7
                for i in range(min(15,self.os.world.size)):
                    row=""
                    for j in range(min(30,self.os.world.size)):
                        row+="." if self.os.world.grid[i][j] is None else "A"
                    try:
                        std.addstr(offset+i,0,row)
                    except curses.error:
                        pass
                std.refresh()
                time.sleep(0.5)
        except curses.error:
            pass
                                                                                            
class AmrOS:
    def __init__(self):
        self.memory=MemoryEngine()
        self.graph=KnowledgeGraph()
        self.net=NeuralNet([5,12,8,2])
        self.emotions=EmotionSystem()
        self.growth=GrowthSystem()
        self.goals=GoalSystem()
        self.wealth=WealthEngine(self.growth,self.emotions)
        self.strategy=StrategyEngine(self.net)
        self.world=World(size=20)
        for _ in range(10):
            self.world.add_agent(Agent(self.world))
        self.world_evo=WorldEvolution(self.world)
        self.loop=TrainingLoop(self.net,self.emotions,self.growth,self.wealth,self.world)
        self.logger=Logger()
        self.vm=StackVM()
        self.compiler=ScriptCompiler()
        EmotionNode(self.emotions).start()
        integrate(self)
                                                            
    def train(self):
        try:
            self.loop.step()
            self.world_evo.evolve()
            self.logger.log("Training step executed")
        except:
            pass
                
    def run_script(self, code):
        bytecode=self.compiler.compile(code)
        self.vm.run(bytecode)
                
    def boot(self):
        print("="*60)
        print(f"{VERSION} booting...")
        print(f"Built by {CREATOR}")
        print("Initializing world, economy, cognition...")
        time.sleep(1)
        print("System ready!")
        print("="*60)
                
    def repl(self):
        self.boot()
        print("Welcome to AMR-OS Teen Edition")
        while True:
            cmd=input("amr-os> ")
            if cmd=="exit":
                break
            elif cmd=="train":
                self.train()
            elif cmd=="wealth":
                try:
                    self.wealth.simulate_trade()
                except:
                    pass
            elif cmd.startswith("goal"):
                self.goals.add(cmd[5:])
            elif cmd=="nextgoal":
                self.goals.next_goal()
            elif cmd.startswith("remember"):
                self.memory.store(cmd[9:])
            elif cmd.startswith("recall"):
                print(self.memory.search(cmd[7:]))
            elif cmd=="dashboard":
                Dashboard(self).start()
            elif cmd.startswith("script"):
                lines=[]
                while True:
                    l=input()
                    if l.strip()=="END": break
                    lines.append(l)
                code="\n".join(lines)
                self.run_script(code)
            elif cmd=="status":
                print("Emotion:",self.emotions.state)
                print("Growth:",self.growth.metrics)
                print("Wealth:",self.wealth.capital)
            else:
                print("Unknown command")
                                                            
if __name__=="__main__":
    AmrOS().repl()