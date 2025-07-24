
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class SingletonExample(metaclass=SingletonMeta):
    def __init__(self):
        self.data = {}

class Observer:
    def __init__(self):
        self.subscribers = []
    def subscribe(self, callback):
        self.subscribers.append(callback)
    def notify(self, *args, **kwargs):
        for callback in self.subscribers:
            callback(*args, **kwargs)

class EventBus:
    def __init__(self):
        self.listeners = {}
    def register(self, event, callback):
        self.listeners.setdefault(event, []).append(callback)
    def emit(self, event, *args):
        for cb in self.listeners.get(event, []):
            cb(*args)

class Config:
    def __init__(self, settings=None):
        self.settings = settings or {}
    def get(self, key, default=None):
        return self.settings.get(key, default)

class Logger:
    def __init__(self):
        self.logs = []
    def log(self, message):
        self.logs.append(message)

class DatabaseConnection:
    def __init__(self, uri):
        self.uri = uri
        self.connected = False
    def connect(self):
        self.connected = True

class Repository:
    def __init__(self):
        self.data = []
    def add(self, item):
        self.data.append(item)
    def all(self):
        return self.data

class Cache:
    def __init__(self):
        self.store = {}
    def set(self, key, value):
        self.store[key] = value
    def get(self, key):
        return self.store.get(key)

class SkipListNode:
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    def __init__(self, max_level, p):
        self.max_level = max_level
        self.p = p
        self.header = SkipListNode(-1, max_level)
        self.level = 0

class Command:
    def execute(self):
        pass

class MacroCommand(Command):
    def __init__(self):
        self.commands = []
    def add(self, command):
        self.commands.append(command)
    def execute(self):
        for c in self.commands:
            c.execute()

class Adapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

class Facade:
    def __init__(self):
        self.subsystems = []
    def operation(self):
        return "Facade operation"

class Proxy:
    def __init__(self, real_subject):
        self._real_subject = real_subject
    def request(self):
        return self._real_subject.request()

class State:
    def handle(self):
        pass

class Context:
    def __init__(self, state):
        self._state = state
    def request(self):
        self._state.handle()

class Strategy:
    def execute(self):
        pass

class ContextStrategy:
    def __init__(self, strategy):
        self._strategy = strategy
    def execute(self):
        return self._strategy.execute()

# ===== ADDITIONAL DESIGN PATTERNS =====

# Factory Pattern
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown animal type: {animal_type}")

# Abstract Factory Pattern
class AbstractFactory:
    def create_product_a(self):
        pass
    def create_product_b(self):
        pass

class ConcreteFactory1(AbstractFactory):
    def create_product_a(self):
        return "ProductA1"
    def create_product_b(self):
        return "ProductB1"

# Builder Pattern
class Computer:
    def __init__(self):
        self.parts = []
    def add(self, part):
        self.parts.append(part)
    def list_parts(self):
        return f"Computer parts: {', '.join(self.parts)}"

class ComputerBuilder:
    def __init__(self):
        self.reset()
    def reset(self):
        self._computer = Computer()
    def build_cpu(self):
        self._computer.add("CPU")
        return self
    def build_memory(self):
        self._computer.add("Memory")
        return self
    def build_disk(self):
        self._computer.add("Disk")
        return self
    def get_result(self):
        result = self._computer
        self.reset()
        return result

# Prototype Pattern
import copy

class Prototype:
    def clone(self):
        return copy.deepcopy(self)

class Document(Prototype):
    def __init__(self, content=""):
        self.content = content
    def clone(self):
        return Document(self.content)

# Bridge Pattern
class Implementation:
    def operation_implementation(self):
        pass

class ConcreteImplementationA(Implementation):
    def operation_implementation(self):
        return "ConcreteImplementationA"

class Abstraction:
    def __init__(self, implementation):
        self._implementation = implementation
    def operation(self):
        return self._implementation.operation_implementation()

# Composite Pattern
class Component:
    def operation(self):
        pass

class Leaf(Component):
    def __init__(self, name):
        self.name = name
    def operation(self):
        return f"Leaf {self.name}"

class Composite(Component):
    def __init__(self):
        self.children = []
    def add(self, component):
        self.children.append(component)
    def operation(self):
        results = []
        for child in self.children:
            results.append(child.operation())
        return f"Composite({', '.join(results)})"

# Decorator Pattern
class Coffee:
    def cost(self):
        return 10
    def description(self):
        return "Simple coffee"

class CoffeeDecorator(Coffee):
    def __init__(self, coffee):
        self._coffee = coffee
    def cost(self):
        return self._coffee.cost()
    def description(self):
        return self._coffee.description()

class Milk(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 2
    def description(self):
        return self._coffee.description() + ", milk"

class Sugar(CoffeeDecorator):
    def cost(self):
        return self._coffee.cost() + 1
    def description(self):
        return self._coffee.description() + ", sugar"

# Flyweight Pattern
class Flyweight:
    def operation(self, extrinsic_state):
        pass

class ConcreteFlyweight(Flyweight):
    def __init__(self, intrinsic_state):
        self._intrinsic_state = intrinsic_state
    def operation(self, extrinsic_state):
        return f"ConcreteFlyweight: {self._intrinsic_state}, {extrinsic_state}"

class FlyweightFactory:
    def __init__(self):
        self._flyweights = {}
    def get_flyweight(self, key):
        if key not in self._flyweights:
            self._flyweights[key] = ConcreteFlyweight(key)
        return self._flyweights[key]

# Chain of Responsibility Pattern
class Handler:
    def __init__(self):
        self._next_handler = None
    def set_next(self, handler):
        self._next_handler = handler
        return handler
    def handle(self, request):
        if self._next_handler:
            return self._next_handler.handle(request)
        return None

class ConcreteHandlerA(Handler):
    def handle(self, request):
        if request == "A":
            return f"ConcreteHandlerA handled {request}"
        return super().handle(request)

class ConcreteHandlerB(Handler):
    def handle(self, request):
        if request == "B":
            return f"ConcreteHandlerB handled {request}"
        return super().handle(request)

# Iterator Pattern
class Iterator:
    def has_next(self):
        pass
    def next(self):
        pass

class ConcreteIterator(Iterator):
    def __init__(self, collection):
        self._collection = collection
        self._index = 0
    def has_next(self):
        return self._index < len(self._collection)
    def next(self):
        if self.has_next():
            item = self._collection[self._index]
            self._index += 1
            return item
        raise StopIteration

class Collection:
    def create_iterator(self):
        pass

class ConcreteCollection(Collection):
    def __init__(self):
        self._items = []
    def add_item(self, item):
        self._items.append(item)
    def create_iterator(self):
        return ConcreteIterator(self._items)

# Mediator Pattern
class Mediator:
    def notify(self, sender, event):
        pass

class ConcreteMediator(Mediator):
    def __init__(self, component1, component2):
        self._component1 = component1
        self._component2 = component2
        self._component1.mediator = self
        self._component2.mediator = self
    def notify(self, sender, event):
        if event == "A":
            self._component2.react_on_a()
        elif event == "D":
            self._component1.react_on_d()

class BaseComponent:
    def __init__(self, mediator=None):
        self._mediator = mediator
    @property
    def mediator(self):
        return self._mediator
    @mediator.setter
    def mediator(self, mediator):
        self._mediator = mediator

class Component1(BaseComponent):
    def do_a(self):
        self._mediator.notify(self, "A")
    def react_on_d(self):
        return "Component 1 reacts on D"

class Component2(BaseComponent):
    def do_d(self):
        self._mediator.notify(self, "D")
    def react_on_a(self):
        return "Component 2 reacts on A"

# Memento Pattern
class Memento:
    def __init__(self, state):
        self._state = state
    def get_state(self):
        return self._state

class Originator:
    def __init__(self):
        self._state = ""
    def set_state(self, state):
        self._state = state
    def get_state(self):
        return self._state
    def save_state_to_memento(self):
        return Memento(self._state)
    def get_state_from_memento(self, memento):
        self._state = memento.get_state()

class CareTaker:
    def __init__(self):
        self._memento_list = []
    def add(self, state):
        self._memento_list.append(state)
    def get(self, index):
        return self._memento_list[index]

# Template Method Pattern
from abc import ABC, abstractmethod

class AbstractClass(ABC):
    def template_method(self):
        self.base_operation1()
        self.required_operations1()
        self.base_operation2()
        self.hook1()
        self.required_operations2()
        self.base_operation3()
        self.hook2()
    def base_operation1(self):
        return "AbstractClass says: I am doing the bulk of the work"
    def base_operation2(self):
        return "AbstractClass says: But I let subclasses override some operations"
    def base_operation3(self):
        return "AbstractClass says: But I am doing the bulk of the work anyway"
    @abstractmethod
    def required_operations1(self):
        pass
    @abstractmethod
    def required_operations2(self):
        pass
    def hook1(self):
        pass
    def hook2(self):
        pass

class ConcreteClass1(AbstractClass):
    def required_operations1(self):
        return "ConcreteClass1 says: Implemented Operation1"
    def required_operations2(self):
        return "ConcreteClass1 says: Implemented Operation2"

class ConcreteClass2(AbstractClass):
    def required_operations1(self):
        return "ConcreteClass2 says: Implemented Operation1"
    def required_operations2(self):
        return "ConcreteClass2 says: Implemented Operation2"
    def hook1(self):
        return "ConcreteClass2 says: Overridden Hook1"

# Visitor Pattern
class Visitor:
    def visit_concrete_component_a(self, element):
        pass
    def visit_concrete_component_b(self, element):
        pass

class ConcreteVisitor1(Visitor):
    def visit_concrete_component_a(self, element):
        return f"{element.exclusive_method_of_concrete_component_a()} + ConcreteVisitor1"
    def visit_concrete_component_b(self, element):
        return f"{element.special_method_of_concrete_component_b()} + ConcreteVisitor1"

class Component:
    def accept(self, visitor):
        pass

class ConcreteComponentA(Component):
    def accept(self, visitor):
        visitor.visit_concrete_component_a(self)
    def exclusive_method_of_concrete_component_a(self):
        return "A"

class ConcreteComponentB(Component):
    def accept(self, visitor):
        visitor.visit_concrete_component_b(self)
    def special_method_of_concrete_component_b(self):
        return "B"

# ===== DATA STRUCTURES =====

# Binary Tree
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None
    def insert(self, value):
        if not self.root:
            self.root = TreeNode(value)
        else:
            self._insert_recursive(self.root, value)
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                self._insert_recursive(node.right, value)

# Graph
class Graph:
    def __init__(self):
        self.vertices = {}
    def add_vertex(self, vertex):
        if vertex not in self.vertices:
            self.vertices[vertex] = []
    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.vertices and vertex2 in self.vertices:
            self.vertices[vertex1].append(vertex2)
            self.vertices[vertex2].append(vertex1)

# Stack
class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    def is_empty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)

# Queue
class Queue:
    def __init__(self):
        self.items = []
    def enqueue(self, item):
        self.items.append(item)
    def dequeue(self):
        if not self.is_empty():
            return self.items.pop(0)
        raise IndexError("Queue is empty")
    def front(self):
        if not self.is_empty():
            return self.items[0]
        raise IndexError("Queue is empty")
    def is_empty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)

# Priority Queue
class PriorityQueue:
    def __init__(self):
        self.queue = []
    def enqueue(self, item, priority):
        self.queue.append((priority, item))
        self.queue.sort(reverse=True)
    def dequeue(self):
        if not self.is_empty():
            return self.queue.pop(0)[1]
        raise IndexError("Priority queue is empty")
    def is_empty(self):
        return len(self.queue) == 0

# ===== ALGORITHM PATTERNS =====

# Sorting Algorithms
class Sorter:
    @staticmethod
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr
    
    @staticmethod
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return Sorter.quick_sort(left) + middle + Sorter.quick_sort(right)

# Search Algorithms
class Searcher:
    @staticmethod
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    @staticmethod
    def linear_search(arr, target):
        for i, item in enumerate(arr):
            if item == target:
                return i
        return -1

# ===== CONCURRENCY PATTERNS =====

# Thread Pool
import threading
import queue
import time

class ThreadPool:
    def __init__(self, num_threads):
        self.tasks = queue.Queue()
        self.threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=self._worker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
    
    def _worker(self):
        while True:
            task = self.tasks.get()
            if task is None:
                break
            task()
            self.tasks.task_done()
    
    def submit(self, task):
        self.tasks.put(task)
    
    def shutdown(self):
        for _ in self.threads:
            self.tasks.put(None)
        for thread in self.threads:
            thread.join()

# ===== ARCHITECTURAL PATTERNS =====

# MVC Pattern
class Model:
    def __init__(self):
        self.data = []
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify_observers(self):
        for observer in self.observers:
            observer.update(self.data)
    
    def add_item(self, item):
        self.data.append(item)
        self.notify_observers()
    
    def get_data(self):
        return self.data

class View:
    def __init__(self, model):
        self.model = model
        self.model.add_observer(self)
    
    def update(self, data):
        print(f"View updated with data: {data}")
    
    def display(self):
        print(f"Current data: {self.model.get_data()}")

class Controller:
    def __init__(self, model, view):
        self.model = model
        self.view = view
    
    def add_item(self, item):
        self.model.add_item(item)
    
    def display_data(self):
        self.view.display()

# ===== BEHAVIORAL PATTERNS =====

# Chain of Responsibility with Middleware
class Middleware:
    def __init__(self):
        self.next_middleware = None
    
    def set_next(self, middleware):
        self.next_middleware = middleware
        return middleware
    
    def process(self, request):
        if self.next_middleware:
            return self.next_middleware.process(request)
        return None

class AuthenticationMiddleware(Middleware):
    def process(self, request):
        if "token" in request:
            print("Authentication passed")
            return super().process(request)
        else:
            return "Authentication failed"

class LoggingMiddleware(Middleware):
    def process(self, request):
        print(f"Logging request: {request}")
        return super().process(request)

# ===== STRUCTURAL PATTERNS =====

# Adapter with Multiple Interfaces
class OldSystem:
    def old_method(self):
        return "Old system method"

class NewSystem:
    def new_method(self):
        return "New system method"

class SystemAdapter:
    def __init__(self, old_system, new_system):
        self.old_system = old_system
        self.new_system = new_system
    
    def unified_method(self):
        return f"{self.old_system.old_method()} + {self.new_system.new_method()}"

# ===== CREATIONAL PATTERNS =====

# Object Pool
class ObjectPool:
    def __init__(self, create_func, max_size=10):
        self.create_func = create_func
        self.max_size = max_size
        self.pool = []
        self.in_use = set()
    
    def acquire(self):
        if self.pool:
            obj = self.pool.pop()
        else:
            obj = self.create_func()
        self.in_use.add(obj)
        return obj
    
    def release(self, obj):
        if obj in self.in_use:
            self.in_use.remove(obj)
            if len(self.pool) < self.max_size:
                self.pool.append(obj)

# ===== UTILITY PATTERNS =====

# Configuration Manager
class ConfigManager:
    def __init__(self):
        self.config = {}
        self._load_default_config()
    
    def _load_default_config(self):
        self.config = {
            'debug': False,
            'port': 8080,
            'host': 'localhost',
            'timeout': 30
        }
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
    
    def load_from_file(self, filename):
        # Simulate loading from file
        pass
    
    def save_to_file(self, filename):
        # Simulate saving to file
        pass

# Connection Pool
class ConnectionPool:
    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self.connections = []
        self.available = []
    
    def get_connection(self):
        if self.available:
            return self.available.pop()
        elif len(self.connections) < self.max_connections:
            conn = self._create_connection()
            self.connections.append(conn)
            return conn
        else:
            raise Exception("No available connections")
    
    def return_connection(self, connection):
        if connection in self.connections:
            self.available.append(connection)
    
    def _create_connection(self):
        return f"Connection_{len(self.connections) + 1}"

# ===== TESTING PATTERNS =====

# Mock Object
class MockDatabase:
    def __init__(self):
        self.data = {}
        self.calls = []
    
    def insert(self, table, data):
        self.calls.append(('insert', table, data))
        self.data[table] = self.data.get(table, []) + [data]
        return True
    
    def select(self, table, conditions=None):
        self.calls.append(('select', table, conditions))
        return self.data.get(table, [])
    
    def get_calls(self):
        return self.calls

# Test Double
class TestDouble:
    def __init__(self):
        self.expected_calls = []
        self.actual_calls = []
    
    def expect_call(self, method_name, args=None, return_value=None):
        self.expected_calls.append({
            'method': method_name,
            'args': args,
            'return_value': return_value
        })
    
    def call(self, method_name, args=None):
        self.actual_calls.append({
            'method': method_name,
            'args': args
        })
        
        for expected in self.expected_calls:
            if expected['method'] == method_name:
                return expected['return_value']
        return None
    
    def verify(self):
        return len(self.expected_calls) == len(self.actual_calls)

# ===== PATTERN COMBINATIONS =====

# Composite + Visitor
class FileSystemComponent:
    def accept(self, visitor):
        pass

class File(FileSystemComponent):
    def __init__(self, name, size):
        self.name = name
        self.size = size
    
    def accept(self, visitor):
        visitor.visit_file(self)

class Directory(FileSystemComponent):
    def __init__(self, name):
        self.name = name
        self.children = []
    
    def add(self, component):
        self.children.append(component)
    
    def accept(self, visitor):
        visitor.visit_directory(self)
        for child in self.children:
            child.accept(visitor)

class FileSystemVisitor:
    def visit_file(self, file):
        print(f"File: {file.name}, Size: {file.size}")
    
    def visit_directory(self, directory):
        print(f"Directory: {directory.name}")

# Factory + Singleton
class DatabaseFactory(metaclass=SingletonMeta):
    def create_database(self, db_type):
        if db_type == "mysql":
            return MySQLDatabase()
        elif db_type == "postgresql":
            return PostgreSQLDatabase()
        else:
            raise ValueError(f"Unknown database type: {db_type}")

class MySQLDatabase:
    def connect(self):
        return "Connected to MySQL"

class PostgreSQLDatabase:
    def connect(self):
        return "Connected to PostgreSQL"

# ===== ADVANCED PATTERNS =====

# Dependency Injection Container
class Container:
    def __init__(self):
        self.services = {}
        self.singletons = {}
    
    def register(self, service_type, implementation=None):
        if implementation is None:
            implementation = service_type
        self.services[service_type] = implementation
    
    def resolve(self, service_type):
        if service_type in self.singletons:
            return self.singletons[service_type]
        
        if service_type not in self.services:
            raise Exception(f"Service {service_type} not registered")
        
        implementation = self.services[service_type]
        instance = implementation()
        
        # Check if it's a singleton
        if hasattr(implementation, '__singleton__'):
            self.singletons[service_type] = instance
        
        return instance

# Event Sourcing
class Event:
    def __init__(self, event_type, data):
        self.event_type = event_type
        self.data = data
        self.timestamp = time.time()

class EventStore:
    def __init__(self):
        self.events = []
    
    def append(self, event):
        self.events.append(event)
    
    def get_events(self, aggregate_id):
        return [e for e in self.events if e.data.get('aggregate_id') == aggregate_id]

class Aggregate:
    def __init__(self, aggregate_id):
        self.aggregate_id = aggregate_id
        self.version = 0
    
    def apply(self, event):
        self.version += 1
        # Apply event to aggregate state
        pass

# ===== FUNCTIONAL PATTERNS =====

# Monad-like Pattern
class Maybe:
    def __init__(self, value):
        self.value = value
    
    def bind(self, func):
        if self.value is None:
            return Maybe(None)
        try:
            result = func(self.value)
            return Maybe(result)
        except:
            return Maybe(None)
    
    def get_or_else(self, default):
        return self.value if self.value is not None else default

# Functor Pattern
class Functor:
    def __init__(self, value):
        self.value = value
    
    def map(self, func):
        return Functor(func(self.value))
    
    def __str__(self):
        return f"Functor({self.value})"

# ===== METAPROGRAMMING PATTERNS =====

# Class Decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Database:
    def __init__(self):
        self.connection = "Connected"

# Method Decorator
def retry(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed, retrying...")
            return None
        return wrapper
    return decorator

class Service:
    @retry(max_attempts=3)
    def unreliable_method(self):
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success"

# ===== CONCURRENCY PATTERNS =====

# Producer-Consumer
class ProducerConsumer:
    def __init__(self, buffer_size=10):
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.producers = []
        self.consumers = []
    
    def add_producer(self, producer_func):
        def producer():
            while True:
                item = producer_func()
                self.buffer.put(item)
        self.producers.append(producer)
    
    def add_consumer(self, consumer_func):
        def consumer():
            while True:
                item = self.buffer.get()
                consumer_func(item)
                self.buffer.task_done()
        self.consumers.append(consumer)

# ===== CACHING PATTERNS =====

# LRU Cache
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# ===== VALIDATION PATTERNS =====

# Validator Chain
class Validator:
    def __init__(self):
        self.next_validator = None
    
    def set_next(self, validator):
        self.next_validator = validator
        return validator
    
    def validate(self, data):
        if self.next_validator:
            return self.next_validator.validate(data)
        return True

class EmailValidator(Validator):
    def validate(self, data):
        if '@' not in data.get('email', ''):
            return False
        return super().validate(data)

class AgeValidator(Validator):
    def validate(self, data):
        age = data.get('age', 0)
        if age < 18 or age > 100:
            return False
        return super().validate(data)

# ===== ERROR HANDLING PATTERNS =====

# Result Pattern
class Result:
    def __init__(self, success, value=None, error=None):
        self.success = success
        self.value = value
        self.error = error
    
    @staticmethod
    def success(value):
        return Result(True, value=value)
    
    @staticmethod
    def failure(error):
        return Result(False, error=error)
    
    def map(self, func):
        if self.success:
            try:
                return Result.success(func(self.value))
            except Exception as e:
                return Result.failure(str(e))
        return self
    
    def flat_map(self, func):
        if self.success:
            return func(self.value)
        return self

# ===== CONFIGURATION PATTERNS =====

# Builder with Fluent Interface
class QueryBuilder:
    def __init__(self):
        self.query = {}
    
    def select(self, *fields):
        self.query['select'] = fields
        return self
    
    def from_table(self, table):
        self.query['from'] = table
        return self
    
    def where(self, condition):
        self.query['where'] = condition
        return self
    
    def order_by(self, field, direction='ASC'):
        self.query['order_by'] = (field, direction)
        return self
    
    def build(self):
        return self.query

# ===== TESTING UTILITIES =====

# Test Data Builder
class UserBuilder:
    def __init__(self):
        self.user = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'age': 30
        }
    
    def with_name(self, name):
        self.user['name'] = name
        return self
    
    def with_email(self, email):
        self.user['email'] = email
        return self
    
    def with_age(self, age):
        self.user['age'] = age
        return self
    
    def build(self):
        return self.user.copy()

# ===== SUMMARY =====
"""
This file demonstrates over 50 different Object-Oriented patterns and classes in Python:

CREATIONAL PATTERNS:
- Singleton (with metaclass)
- Factory Method
- Abstract Factory
- Builder
- Prototype
- Object Pool

STRUCTURAL PATTERNS:
- Adapter
- Bridge
- Composite
- Decorator
- Facade
- Flyweight
- Proxy

BEHAVIORAL PATTERNS:
- Chain of Responsibility
- Command
- Iterator
- Mediator
- Memento
- Observer
- State
- Strategy
- Template Method
- Visitor

ARCHITECTURAL PATTERNS:
- MVC
- Repository
- Event Sourcing
- Dependency Injection

DATA STRUCTURES:
- Binary Tree
- Graph
- Stack
- Queue
- Priority Queue
- Skip List

ALGORITHM PATTERNS:
- Sorting (Bubble, Quick)
- Searching (Binary, Linear)

CONCURRENCY PATTERNS:
- Thread Pool
- Producer-Consumer

CACHING PATTERNS:
- LRU Cache
- Simple Cache

VALIDATION PATTERNS:
- Validator Chain

ERROR HANDLING PATTERNS:
- Result Pattern

UTILITY PATTERNS:
- Configuration Manager
- Connection Pool
- Event Bus
- Logger

TESTING PATTERNS:
- Mock Objects
- Test Doubles
- Test Data Builders

FUNCTIONAL PATTERNS:
- Maybe Monad
- Functor

METAPROGRAMMING PATTERNS:
- Class Decorators
- Method Decorators
- Metaclasses

This demonstrates the incredible flexibility and power of Python's object-oriented programming capabilities!
"""
