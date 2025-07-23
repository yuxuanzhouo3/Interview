太好了！我马上开始给你第1批10个系统设计题目，包含系统说明、关键设计点、核心类设计（Python示例）和关键代码片段。

---

# 系统设计题目1–10

---

### 1. **设计一个简易的URL短链接服务**

** 系统简介 **
将长URL转成短链接，用户访问短链接时重定向到长链接。

** 关键设计点 **

*唯一短码生成（Base62编码、自增ID映射）
*存储映射（短码 → 长URL）
*高可用和扩展
*缓存热点短链接

** 核心类设计 **

```python
import hashlib


class URLShortener:
    def __init__(self):
        self.url_map = {}
        self.counter = 1
        self.base62 = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def encode(self, num):
        s = []
        while num > 0:
            s.append(self.base62[num % 62])
            num //= 62
        return "".join(reversed(s))

    def shorten(self, long_url):
        short_key = self.encode(self.counter)
        self.url_map[short_key] = long_url
        self.counter += 1
        return short_key

    def retrieve(self, short_key):
        return self.url_map.get(short_key, None)


```

---

### 2. **设计一个聊天室系统**

** 系统简介 **
支持多用户，实时消息广播，消息历史。

** 关键设计点 **

*消息队列（消息广播）
*用户连接管理（WebSocket）
*消息持久化存储
*在线状态管理

** 核心类设计 **

```python


class ChatRoom:
    def __init__(self):
        self.users = set()
        self.messages = []

    def join(self, user):
        self.users.add(user)

    def leave(self, user):
        self.users.discard(user)

    def broadcast(self, message):
        self.messages.append(message)
        for user in self.users:
            user.receive(message)


```

---

### 3. **设计一个简易的文件存储服务（类似Google Drive）**

** 系统简介 **
用户上传、下载、管理文件。

** 关键设计点 **

*文件存储（分布式文件系统或对象存储）
*元数据管理（文件名、路径、权限）
*访问控制
*分片上传和断点续传

** 核心类设计 **

```python


class FileStorage:
    def __init__(self):
        self.files = {}  # filename -> content

    def upload(self, filename, content):
        self.files[filename] = content

    def download(self, filename):
        return self.files.get(filename)

    def delete(self, filename):
        if filename in self.files:
            del self.files[filename]


```

---

### 4. **设计一个在线图片识别系统**

** 系统简介 **
用户上传图片，系统返回识别结果。

** 关键设计点 **

*图片接收与预处理
*AI模型推理接口（异步任务队列）
*结果缓存
*高并发处理

** 核心类设计 **

```python


class ImageRecognitionService:
    def __init__(self, model):
        self.model = model

    def recognize(self, image):
        processed = self.preprocess(image)
        result = self.model.predict(processed)
        return result

    def preprocess(self, image):
        # 图像预处理代码
        return image


```

---

### 5. **设计一个分布式锁系统**

** 系统简介 **
多个客户端通过分布式锁协调访问共享资源。

** 关键设计点 **

*锁获取与释放
*锁超时防止死锁
*锁的可重入性
*高性能和可用性

** 核心类设计 **

```python
import threading
import time


class DistributedLock:
    def __init__(self):
        self.lock_map = {}
        self.lock = threading.Lock()

    def acquire(self, key, timeout=10):
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if key not in self.lock_map:
                    self.lock_map[key] = True
                    return True
            time.sleep(0.01)
        return False

    def release(self, key):
        with self.lock:
            if key in self.lock_map:
                del self.lock_map[key]


```

---

### 6. **设计一个推荐系统（简化版）**

** 系统简介 **
根据用户历史行为推荐商品。

** 关键设计点 **

*用户行为数据采集
*推荐算法（协同过滤）
*实时推荐和离线训练分离
*缓存热门推荐结果

** 核心类设计 **

```python


class Recommender:
    def __init__(self):
        self.user_items = {}  # userId -> set of itemIds

    def add_interaction(self, user_id, item_id):
        self.user_items.setdefault(user_id, set()).add(item_id)

    def recommend(self, user_id):
        # 简单协同过滤示例
        interacted = self.user_items.get(user_id, set())
        candidates = {}
        for other_user, items in self.user_items.items():
            if other_user == user_id: continue
            for item in items - interacted:
                candidates[item] = candidates.get(item, 0) + 1
        return sorted(candidates.keys(), key=lambda x: -candidates[x])[:5]


```

---

### 7. **设计一个简易的分布式消息队列**

** 系统简介 **
消息生产者发送消息，消费者异步消费。

** 关键设计点 **

*消息持久化
*消息分发策略
*消费者确认机制
*消息重复与顺序保证

** 核心类设计 **

```python
from collections import deque


class MessageQueue:
    def __init__(self):
        self.queue = deque()

    def publish(self, message):
        self.queue.append(message)

    def consume(self):
        if self.queue:
            return self.queue.popleft()
        return None


```

---

### 8. **设计一个基于时间的Key-Value存储**

** 系统简介 **
支持根据时间戳查询某个key的值。

** 关键设计点 **

*多版本存储
*二分查找历史数据
*数据压缩

** 核心类设计 **

```python
from collections import defaultdict
import bisect


class TimeMap:
    def __init__(self):
        self.store = defaultdict(list)  # key -> [(timestamp, value)]

    def set(self, key, value, timestamp):
        self.store[key].append((timestamp, value))

    def get(self, key, timestamp):
        arr = self.store.get(key, [])
        i = bisect.bisect(arr, (timestamp, chr(127)))
        return arr[i - 1][1] if i else ""


```

---

### 9. **设计一个任务调度系统**

** 系统简介 **
支持定时执行任务，周期性任务调度。

** 关键设计点 **

*定时器管理
*任务队列和执行线程
*失败重试机制

** 核心类设计 **

```python
import threading
import time
import heapq


class TaskScheduler:
    def __init__(self):
        self.tasks = []
        self.cv = threading.Condition()
        threading.Thread(target=self.run, daemon=True).start()

    def schedule(self, delay, func):
        with self.cv:
            run_at = time.time() + delay
            heapq.heappush(self.tasks, (run_at, func))
            self.cv.notify()

    def run(self):
        while True:
            with self.cv:
                while not self.tasks or self.tasks[0][0] > time.time():
                    if self.tasks:
                        timeout = self.tasks[0][0] - time.time()
                        self.cv.wait(timeout)
                    else:
                        self.cv.wait()
                run_at, func = heapq.heappop(self.tasks)
            try:
                func()
            except Exception:
                pass


```

---

### 10. **设计一个在线考试系统**

** 系统简介 **
支持考生登录、答题、交卷和评分。

** 关键设计点 **

*用户身份验证
*题库管理
*自动评分和手动评分
*防作弊机制

** 核心类设计 **

```python


class ExamSystem:
    def __init__(self):
        self.users = set()
        self.questions = {}
        self.answers = {}

    def register(self, user_id):
        self.users.add(user_id)

    def add_question(self, q_id, correct_answer):
        self.questions[q_id] = correct_answer

    def submit_answer(self, user_id, q_id, answer):
        self.answers.setdefault(user_id, {})[q_id] = answer

    def grade(self, user_id):
        score = 0
        for q, correct in self.questions.items():
            if self.answers.get(user_id, {}).get(q) == correct:
                score += 1
        return score
```

好的，接着来第二批系统设计题目（11-20），包含系统简介、关键设计点、核心类设计及关键代码示例：

---

# 系统设计题目11–20

---

### 11. **设计一个微博/推特式的消息流系统**

**系统简介**
用户发布短消息，粉丝可看到更新的消息流。

**关键设计点**

* 关注关系存储
* 消息推送策略（Pull vs Push）
* 时间线合并排序
* 缓存热点用户消息

**核心类设计**

```python
import heapq
from collections import defaultdict

class Twitter:
    def __init__(self):
        self.time = 0
        self.tweets = defaultdict(list)  # userId -> [(time, tweetId)]
        self.followers = defaultdict(set)  # userId -> set of userIds
    
    def post_tweet(self, userId, tweetId):
        self.time += 1
        self.tweets[userId].append((self.time, tweetId))
    
    def follow(self, followerId, followeeId):
        self.followers[followerId].add(followeeId)
    
    def unfollow(self, followerId, followeeId):
        self.followers[followerId].discard(followeeId)
    
    def get_news_feed(self, userId):
        heap = []
        users = self.followers[userId] | {userId}
        for u in users:
            for t in self.tweets[u][-10:]:
                heapq.heappush(heap, t)
                if len(heap) > 10:
                    heapq.heappop(heap)
        return [tweetId for _, tweetId in sorted(heap, reverse=True)]
```

---

### 12. **设计一个简易的电商购物车系统**

**系统简介**
用户管理商品加入购物车，调整数量，结算。

**关键设计点**

* 用户购物车状态存储（会话持久化）
* 商品库存同步
* 购物车过期清理
* 多设备同步

**核心类设计**

```python
class ShoppingCart:
    def __init__(self):
        self.cart = {}  # productId -> quantity
    
    def add_item(self, product_id, quantity=1):
        self.cart[product_id] = self.cart.get(product_id, 0) + quantity
    
    def remove_item(self, product_id, quantity=1):
        if product_id in self.cart:
            self.cart[product_id] -= quantity
            if self.cart[product_id] <= 0:
                del self.cart[product_id]
    
    def get_items(self):
        return self.cart
```

---

### 13. **设计一个视频点播系统**

**系统简介**
用户上传、点播视频，支持多码率和缓存。

**关键设计点**

* 视频存储和分发（CDN）
* 转码处理（多码率）
* 播放权限控制
* 断点续播和缓存策略

**核心类设计**

```python
class VideoService:
    def __init__(self):
        self.videos = {}  # videoId -> video metadata
    
    def upload_video(self, video_id, metadata):
        self.videos[video_id] = metadata
    
    def get_video(self, video_id, bitrate='720p'):
        # 返回适合码率的视频地址或流
        return f"url_for_{video_id}_{bitrate}"
```

---

### 14. **设计一个实时竞价广告系统**

**系统简介**
广告主竞价实时展示广告给用户。

**关键设计点**

* 实时竞价决策系统
* 广告库存管理
* 延迟极低的响应时间
* 竞价算法和预算控制

**核心类设计**

```python
class AdBidder:
    def __init__(self):
        self.ads = {}  # adId -> bidPrice
    
    def register_ad(self, ad_id, bid_price):
        self.ads[ad_id] = bid_price
    
    def bid(self, user_context):
        # 选出出价最高的广告
        if not self.ads:
            return None
        return max(self.ads.items(), key=lambda x: x[1])[0]
```

---

### 15. **设计一个在线多人协作文档系统**

**系统简介**
支持多人同时编辑同一文档，实时同步。

**关键设计点**

* 操作冲突处理（OT或CRDT算法）
* 实时消息广播和同步
* 文档版本管理
* 离线编辑和合并

**核心类设计**

```python
class CollaborativeDoc:
    def __init__(self):
        self.content = ""
        self.operations = []
    
    def apply_operation(self, op):
        # 简单操作示例: op = {"pos": int, "insert": str}
        pos, insert_str = op['pos'], op['insert']
        self.content = self.content[:pos] + insert_str + self.content[pos:]
        self.operations.append(op)
    
    def get_content(self):
        return self.content
```

---

### 16. **设计一个新闻推送系统**

**系统简介**
将新闻推送给订阅用户。

**关键设计点**

* 用户订阅管理
* 新闻分类和过滤
* 推送策略（实时 vs 批量）
* 多渠道推送（App通知、邮件）

**核心类设计**

```python
class NewsPushSystem:
    def __init__(self):
        self.subscribers = {}  # userId -> set of categories
    
    def subscribe(self, user_id, category):
        self.subscribers.setdefault(user_id, set()).add(category)
    
    def push_news(self, category, news):
        for user_id, cats in self.subscribers.items():
            if category in cats:
                self.send_to_user(user_id, news)
    
    def send_to_user(self, user_id, news):
        print(f"Pushing news to {user_id}: {news}")
```

---

### 17. **设计一个电子邮件服务系统**

**系统简介**
用户发送、接收和管理电子邮件。

**关键设计点**

* 邮件发送和接收协议（SMTP/IMAP）
* 邮件存储
* 垃圾邮件过滤
* 附件管理

**核心类设计**

```python
class EmailService:
    def __init__(self):
        self.inbox = {}  # user -> [emails]
    
    def send_email(self, from_user, to_user, subject, content):
        email = {"from": from_user, "subject": subject, "content": content}
        self.inbox.setdefault(to_user, []).append(email)
    
    def get_inbox(self, user):
        return self.inbox.get(user, [])
```

---

### 18. **设计一个分布式文件同步系统**

**系统简介**
跨多设备自动同步文件变化。

**关键设计点**

* 文件版本冲突处理
* 增量同步机制
* 断点续传和差异同步
* 离线支持

**核心类设计**

```python
class FileSync:
    def __init__(self):
        self.file_versions = {}  # filename -> version
    
    def update_file(self, filename, version):
        if filename not in self.file_versions or self.file_versions[filename] < version:
            self.file_versions[filename] = version
            return True  # 表示更新成功
        return False
    
    def get_version(self, filename):
        return self.file_versions.get(filename, 0)
```

---

### 19. **设计一个内容分发网络（CDN）缓存系统**

**系统简介**
缓存静态资源，加速全球访问。

**关键设计点**

* 缓存失效策略（TTL/LRU）
* 边缘节点分布
* 缓存预热和清理
* 请求路由优化

**核心类设计**

```python
import time

class CDNCache:
    def __init__(self, ttl=60):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        entry = self.cache.get(key)
        if not entry:
            return None
        if time.time() - entry['time'] > self.ttl:
            del self.cache[key]
            return None
        return entry['value']
    
    def put(self, key, value):
        self.cache[key] = {'value': value, 'time': time.time()}
```

---

### 20. **设计一个在线投票系统**

**系统简介**
用户创建投票，参与投票并统计结果。

**关键设计点**

* 投票合法性验证
* 多选或单选支持
* 投票结果实时统计
* 防刷票机制

**核心类设计**

```python
class VotingSystem:
    def __init__(self):
        self.votes = {}  # vote_id -> {option: count}
        self.user_votes = {}  # (vote_id, user_id) -> bool
    
    def create_vote(self, vote_id, options):
        self.votes[vote_id] = {opt: 0 for opt in options}
    
    def vote(self, vote_id, user_id, option):
        if (vote_id, user_id) in self.user_votes:
            return False  # 已投票
        if option not in self.votes.get(vote_id, {}):
            return False
        self.votes[vote_id][option] += 1
        self.user_votes[(vote_id, user_id)] = True
        return True
    
    def get_results(self, vote_id):
        return self.votes.get(vote_id, {})
```

---

好的，接着来第三批系统设计题目（21-30），同样包含系统简介、关键设计点、核心类设计及关键代码示例：

---

# 系统设计题目21–30

---

### 21. **设计一个在线问答平台（类似知乎/StackOverflow）**

**系统简介**
用户发布问题、回答和评论，支持点赞和排序。

**关键设计点**

* 问题、回答和评论的存储关系
* 热门排序算法（基于点赞数、时间等）
* 用户行为权限控制
* 搜索功能

**核心类设计**

```python
class QASystem:
    def __init__(self):
        self.questions = {}  # questionId -> {"content": str, "answers": list}
        self.answers = {}    # answerId -> {"content": str, "likes": int}
    
    def post_question(self, question_id, content):
        self.questions[question_id] = {"content": content, "answers": []}
    
    def post_answer(self, question_id, answer_id, content):
        self.answers[answer_id] = {"content": content, "likes": 0}
        self.questions[question_id]["answers"].append(answer_id)
    
    def like_answer(self, answer_id):
        if answer_id in self.answers:
            self.answers[answer_id]["likes"] += 1
    
    def get_top_answers(self, question_id):
        ans = self.questions.get(question_id, {}).get("answers", [])
        return sorted(ans, key=lambda aid: self.answers[aid]["likes"], reverse=True)
```

---

### 22. **设计一个自动化备份系统**

**系统简介**
定期备份用户数据，支持增量备份和恢复。

**关键设计点**

* 备份计划调度
* 增量和全量备份策略
* 备份数据校验与恢复
* 存储管理

**核心类设计**

```python
class BackupSystem:
    def __init__(self):
        self.backups = {}  # timestamp -> backup data
    
    def backup(self, data):
        timestamp = self.get_current_time()
        self.backups[timestamp] = data  # 这里简化存储
    
    def restore(self, timestamp):
        return self.backups.get(timestamp, None)
    
    def get_current_time(self):
        import time
        return int(time.time())
```

---

### 23. **设计一个图片识别服务**

**系统简介**
用户上传图片，系统返回识别结果（如物体、文字等）。

**关键设计点**

* 图像预处理
* 识别模型部署与推理
* 异步任务处理
* 结果缓存和优化

**核心类设计**

```python
class ImageRecognitionService:
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self):
        # 假装加载一个模型
        return lambda image: ["cat", "dog"]
    
    def recognize(self, image):
        # 输入image，返回标签列表
        return self.model(image)
```

---

### 24. **设计一个实时位置共享系统**

**系统简介**
用户分享实时地理位置给好友或群组。

**关键设计点**

* 位置数据频率与传输控制
* 位置权限和隐私保护
* 地图与路径展示
* 实时通信支持

**核心类设计**

```python
class LocationSharing:
    def __init__(self):
        self.locations = {}  # userId -> (lat, lon)
        self.friends = {}    # userId -> set of userIds
    
    def update_location(self, user_id, lat, lon):
        self.locations[user_id] = (lat, lon)
    
    def get_friend_locations(self, user_id):
        return {f: self.locations.get(f) for f in self.friends.get(user_id, set())}
    
    def add_friend(self, user_id, friend_id):
        self.friends.setdefault(user_id, set()).add(friend_id)
```

---

### 25. **设计一个实时聊天系统**

**系统简介**
支持一对一及群组聊天，消息实时传递。

**关键设计点**

* 消息存储和队列系统
* 在线用户状态管理
* 消息顺序保证与去重
* 离线消息推送

**核心类设计**

```python
class ChatSystem:
    def __init__(self):
        self.messages = {}  # chatId -> list of messages
    
    def send_message(self, chat_id, message):
        self.messages.setdefault(chat_id, []).append(message)
    
    def get_messages(self, chat_id, last_n=20):
        return self.messages.get(chat_id, [])[-last_n:]
```

---

### 26. **设计一个在线考试系统**

**系统简介**
学生在线参加考试，提交答案，自动评分。

**关键设计点**

* 题库管理
* 考试流程控制（时间、身份验证）
* 自动评分（选择题和主观题）
* 作弊检测

**核心类设计**

```python
class ExamSystem:
    def __init__(self):
        self.exams = {}  # examId -> {"questions": list, "answers": dict}
    
    def create_exam(self, exam_id, questions):
        self.exams[exam_id] = {"questions": questions, "answers": {}}
    
    def submit_answer(self, exam_id, user_id, answers):
        self.exams[exam_id]["answers"][user_id] = answers
    
    def grade_exam(self, exam_id, user_id):
        # 简单自动评分，假设答案是选择题
        correct = 0
        exam = self.exams[exam_id]
        user_answers = exam["answers"].get(user_id, {})
        for qid, ans in exam["questions"].items():
            if user_answers.get(qid) == ans["correct"]:
                correct += 1
        return correct / len(exam["questions"])
```

---

### 27. **设计一个在线旅游预订系统**

**系统简介**
用户查询和预订机票、酒店等。

**关键设计点**

* 库存同步和锁定
* 订单管理与支付流程
* 多渠道查询聚合
* 取消和改签机制

**核心类设计**

```python
class BookingSystem:
    def __init__(self):
        self.inventory = {}  # resourceId -> available count
    
    def check_availability(self, resource_id):
        return self.inventory.get(resource_id, 0) > 0
    
    def book(self, resource_id):
        if self.check_availability(resource_id):
            self.inventory[resource_id] -= 1
            return True
        return False
    
    def cancel(self, resource_id):
        self.inventory[resource_id] = self.inventory.get(resource_id, 0) + 1
```

---

### 28. **设计一个音乐流媒体服务**

**系统简介**
用户点播音乐，支持歌单和个性推荐。

**关键设计点**

* 音乐文件存储和CDN分发
* 用户播放历史和偏好分析
* 推荐算法
* 实时歌词和专辑展示

**核心类设计**

```python
class MusicService:
    def __init__(self):
        self.songs = {}  # songId -> metadata
    
    def add_song(self, song_id, metadata):
        self.songs[song_id] = metadata
    
    def play_song(self, song_id):
        # 返回歌曲流地址（简化）
        return f"url_to_song_{song_id}"
```

---

### 29. **设计一个汽车共享系统**

**系统简介**
用户租用和归还共享汽车。

**关键设计点**

* 汽车定位和状态管理
* 租用和归还流程
* 价格计算与计费
* 用户评价和信用体系

**核心类设计**

```python
class CarShareSystem:
    def __init__(self):
        self.cars = {}  # carId -> {"location": (lat, lon), "status": "available" or "rented"}
        self.rentals = {}  # userId -> carId
    
    def rent_car(self, user_id, car_id):
        if self.cars.get(car_id, {}).get("status") == "available":
            self.cars[car_id]["status"] = "rented"
            self.rentals[user_id] = car_id
            return True
        return False
    
    def return_car(self, user_id):
        car_id = self.rentals.pop(user_id, None)
        if car_id:
            self.cars[car_id]["status"] = "available"
            return True
        return False
```

---

### 30. **设计一个智能家居控制系统**

**系统简介**
用户远程控制和监控家中智能设备。

**关键设计点**

* 设备状态同步
* 规则引擎自动化控制
* 实时通知
* 多设备兼容

**核心类设计**

```python
class SmartHomeSystem:
    def __init__(self):
        self.devices = {}  # deviceId -> status
    
    def set_device_state(self, device_id, state):
        self.devices[device_id] = state
    
    def get_device_state(self, device_id):
        return self.devices.get(device_id)
```
---

# 系统设计题目31–40

---

### 31. **设计一个文件版本控制系统（类似 Git）**

**系统简介**
支持文件的版本管理，提交、分支和合并。

**关键设计点**

* 快照机制（存储增量或完整快照）
* 分支和合并策略
* 冲突检测与解决
* 元数据管理（提交记录、作者信息）

**核心类设计**

```python
class Commit:
    def __init__(self, id, files, parent=None):
        self.id = id
        self.files = files  # dict: filename -> content
        self.parent = parent

class VersionControlSystem:
    def __init__(self):
        self.commits = {}  # commit_id -> Commit
        self.branches = {"master": None}
        self.current_branch = "master"

    def commit(self, commit_id, files):
        parent = self.branches[self.current_branch]
        new_commit = Commit(commit_id, files, parent)
        self.commits[commit_id] = new_commit
        self.branches[self.current_branch] = new_commit

    def checkout(self, commit_id):
        commit = self.commits.get(commit_id)
        if commit:
            return commit.files
        return None
```

---

### 32. **设计一个实时股票行情系统**

**系统简介**
提供股票实时价格更新和历史查询。

**关键设计点**

* 高并发行情数据推送
* 低延迟数据更新机制
* 历史数据存储与查询
* 多客户端订阅管理

**核心类设计**

```python
class StockTicker:
    def __init__(self):
        self.prices = {}  # stock_symbol -> latest price
        self.subscribers = {}  # stock_symbol -> set of client ids

    def update_price(self, symbol, price):
        self.prices[symbol] = price
        self.notify_subscribers(symbol)

    def subscribe(self, client_id, symbol):
        self.subscribers.setdefault(symbol, set()).add(client_id)

    def notify_subscribers(self, symbol):
        clients = self.subscribers.get(symbol, set())
        for c in clients:
            print(f"Notify {c}: {symbol} price updated to {self.prices[symbol]}")
```

---

### 33. **设计一个电子钱包系统**

**系统简介**
支持用户充值、支付和退款。

**关键设计点**

* 账户余额管理
* 交易流水和状态
* 安全和防止双花攻击
* 支付渠道接口

**核心类设计**

```python
class Wallet:
    def __init__(self):
        self.balances = {}  # user_id -> balance

    def recharge(self, user_id, amount):
        self.balances[user_id] = self.balances.get(user_id, 0) + amount

    def pay(self, user_id, amount):
        if self.balances.get(user_id, 0) >= amount:
            self.balances[user_id] -= amount
            return True
        return False
```

---

### 34. **设计一个新闻推荐系统**

**系统简介**
根据用户兴趣推荐新闻文章。

**关键设计点**

* 用户画像构建
* 内容特征提取
* 推荐算法（协同过滤、内容过滤等）
* 实时更新和反馈机制

**核心类设计**

```python
class NewsRecommender:
    def __init__(self):
        self.user_profiles = {}  # user_id -> set of interests
        self.articles = {}       # article_id -> set of tags

    def add_user_profile(self, user_id, interests):
        self.user_profiles[user_id] = interests

    def add_article(self, article_id, tags):
        self.articles[article_id] = tags

    def recommend(self, user_id):
        interests = self.user_profiles.get(user_id, set())
        recs = []
        for aid, tags in self.articles.items():
            if interests.intersection(tags):
                recs.append(aid)
        return recs
```

---

### 35. **设计一个多人在线协作文档系统**

**系统简介**
多人实时编辑文档，支持版本控制。

**关键设计点**

* 实时同步机制（如 OT 或 CRDT）
* 版本管理
* 冲突解决策略
* 权限管理

**核心类设计**

```python
class CollaborativeDocument:
    def __init__(self):
        self.content = ""
        self.version = 0

    def edit(self, changes):
        # 简单示例: 追加文本
        self.content += changes
        self.version += 1

    def get_content(self):
        return self.content
```

---

### 36. **设计一个内容分发网络（CDN）**

**系统简介**
缓存静态资源，减少源服务器负载。

**关键设计点**

* 缓存策略和失效机制
* 边缘节点管理
* 请求路由和负载均衡
* 缓存预热

**核心类设计**

```python
class CDNNode:
    def __init__(self):
        self.cache = {}

    def get_content(self, url):
        if url in self.cache:
            return self.cache[url]
        content = self.fetch_from_origin(url)
        self.cache[url] = content
        return content

    def fetch_from_origin(self, url):
        return f"Content of {url}"
```

---

### 37. **设计一个在线教育直播平台**

**系统简介**
支持教师直播授课，学生观看和互动。

**关键设计点**

* 视频流传输（低延迟）
* 实时弹幕和问答
* 课程管理
* 直播录制和回放

**核心类设计**

```python
class LiveClass:
    def __init__(self):
        self.students = set()
        self.chat_messages = []

    def join(self, student_id):
        self.students.add(student_id)

    def send_message(self, student_id, message):
        self.chat_messages.append((student_id, message))

    def get_messages(self):
        return self.chat_messages
```

---

### 38. **设计一个自动驾驶车辆控制系统**

**系统简介**
车辆感知环境，规划路径，控制动作。

**关键设计点**

* 传感器数据融合
* 实时路径规划
* 车辆控制接口
* 安全与紧急处理

**核心类设计**

```python
class AutoDriveSystem:
    def __init__(self):
        self.sensor_data = {}

    def update_sensors(self, data):
        self.sensor_data = data

    def plan_path(self):
        # 简化路径规划
        return ["forward", "left", "forward"]

    def execute_commands(self, commands):
        for cmd in commands:
            print(f"Executing {cmd}")
```

---

### 39. **设计一个天气预报系统**

**系统简介**
收集气象数据，预测天气并发布。

**关键设计点**

* 数据采集与处理
* 预测模型集成
* 多维度数据存储
* 实时更新和推送

**核心类设计**

```python
class WeatherSystem:
    def __init__(self):
        self.data = {}

    def update_data(self, location, weather_info):
        self.data[location] = weather_info

    def get_forecast(self, location):
        return self.data.get(location, "No data")
```

---

### 40. **设计一个电子书阅读平台**

**系统简介**
用户在线阅读电子书，支持书签和笔记。

**关键设计点**

* 书籍内容管理
* 阅读进度同步
* 用户书签和笔记存储
* 支持多设备阅读

**核心类设计**

```python
class EBookPlatform:
    def __init__(self):
        self.books = {}          # book_id -> content
        self.user_progress = {}  # user_id -> {book_id: page_number}
        self.user_notes = {}     # user_id -> {book_id: list of notes}

    def read(self, user_id, book_id, page):
        self.user_progress.setdefault(user_id, {})[book_id] = page
        return self.books.get(book_id, "")[page*100:(page+1)*100]

    def add_note(self, user_id, book_id, note):
        self.user_notes.setdefault(user_id, {}).setdefault(book_id, []).append(note)
```

---

# 系统设计题目41–50

---

### 41. **设计一个在线考试系统**

**系统简介**
支持题库管理、考试监控、自动评分。

**关键设计点**

* 题库分类管理
* 实时考试状态管理
* 答案提交和自动评分
* 作弊检测（限制多开、监控行为）

**核心类设计**

```python
class ExamSystem:
    def __init__(self):
        self.exams = {}           # exam_id -> {questions, duration}
        self.submissions = {}     # user_id -> {exam_id -> answers}

    def create_exam(self, exam_id, questions, duration):
        self.exams[exam_id] = {"questions": questions, "duration": duration}

    def submit_answers(self, user_id, exam_id, answers):
        self.submissions.setdefault(user_id, {})[exam_id] = answers

    def grade_exam(self, exam_id, user_id):
        questions = self.exams[exam_id]["questions"]
        answers = self.submissions[user_id][exam_id]
        score = 0
        for q, a in zip(questions, answers):
            if q["correct_answer"] == a:
                score += 1
        return score
```

---

### 42. **设计一个在线打车系统**

**系统简介**
用户呼叫车辆，司机接单，调度系统匹配。

**关键设计点**

* 司机与用户地理位置实时更新
* 匹配算法（最近司机优先）
* 行程状态管理
* 支付与评价系统

**核心类设计**

```python
class RideHailing:
    def __init__(self):
        self.drivers = {}  # driver_id -> location
        self.rides = {}    # ride_id -> details

    def update_driver_location(self, driver_id, location):
        self.drivers[driver_id] = location

    def request_ride(self, user_id, user_location):
        nearest_driver = min(self.drivers.items(),
                             key=lambda d: self.distance(d[1], user_location))
        ride_id = f"ride_{len(self.rides)+1}"
        self.rides[ride_id] = {"user": user_id, "driver": nearest_driver[0], "status": "assigned"}
        return ride_id

    def distance(self, loc1, loc2):
        # 伪代码计算两点距离
        return abs(loc1[0]-loc2[0]) + abs(loc1[1]-loc2[1])
```

---

### 43. **设计一个企业即时通讯系统**

**系统简介**
支持多渠道消息推送，群聊，消息存储。

**关键设计点**

* 实时消息推送（WebSocket）
* 离线消息缓存
* 群聊与私聊
* 消息加密和审计

**核心类设计**

```python
class IMSystem:
    def __init__(self):
        self.users = set()
        self.groups = {}  # group_id -> set of user_ids
        self.messages = {} # user_id -> list of messages

    def send_message(self, sender, receiver, content):
        self.messages.setdefault(receiver, []).append((sender, content))

    def create_group(self, group_id, user_ids):
        self.groups[group_id] = set(user_ids)

    def send_group_message(self, sender, group_id, content):
        for user in self.groups.get(group_id, []):
            if user != sender:
                self.send_message(sender, user, content)
```

---

### 44. **设计一个在线招聘系统**

**系统简介**
发布职位，候选人申请，招聘方筛选。

**关键设计点**

* 职位与简历管理
* 搜索和匹配机制
* 状态跟踪（申请、面试、录用）
* 通知和反馈系统

**核心类设计**

```python
class JobSystem:
    def __init__(self):
        self.jobs = {}         # job_id -> job_info
        self.applications = {} # user_id -> list of job_ids

    def post_job(self, job_id, info):
        self.jobs[job_id] = info

    def apply_job(self, user_id, job_id):
        self.applications.setdefault(user_id, []).append(job_id)

    def get_applications(self, user_id):
        return self.applications.get(user_id, [])
```

---

### 45. **设计一个社交网络的好友推荐系统**

**系统简介**
基于共同好友和兴趣推荐好友。

**关键设计点**

* 社交图建模
* 推荐算法（共同好友数量、兴趣相似度）
* 扩展推荐（好友的好友）
* 推荐结果排序

**核心类设计**

```python
class FriendRecommender:
    def __init__(self):
        self.friends = {}  # user_id -> set of friend_ids
        self.interests = {} # user_id -> set of interests

    def add_friend(self, user_id, friend_id):
        self.friends.setdefault(user_id, set()).add(friend_id)
        self.friends.setdefault(friend_id, set()).add(user_id)

    def recommend(self, user_id):
        candidates = {}
        user_friends = self.friends.get(user_id, set())
        user_interests = self.interests.get(user_id, set())

        for friend in user_friends:
            for fof in self.friends.get(friend, set()):
                if fof != user_id and fof not in user_friends:
                    score = len(self.interests.get(fof, set()).intersection(user_interests)) + 1
                    candidates[fof] = candidates.get(fof, 0) + score

        return sorted(candidates.items(), key=lambda x: -x[1])
```

---

### 46. **设计一个在线视频编辑平台**

**系统简介**
上传视频，剪辑，拼接，特效处理。

**关键设计点**

* 视频文件存储和处理流水线
* 支持多轨道编辑
* 渲染引擎与导出功能
* 用户协作编辑

**核心类设计**

```python
class VideoEditor:
    def __init__(self):
        self.tracks = []  # 每个轨道是视频片段列表

    def add_clip(self, track_id, clip):
        while len(self.tracks) <= track_id:
            self.tracks.append([])
        self.tracks[track_id].append(clip)

    def render(self):
        # 简单合并所有轨道剪辑
        rendered = []
        for track in self.tracks:
            rendered.extend(track)
        return rendered
```

---

### 47. **设计一个云存储系统**

**系统简介**
文件上传、下载、共享、权限管理。

**关键设计点**

* 文件分片与冗余存储
* 权限控制与分享链接
* 版本控制
* 容量与带宽管理

**核心类设计**

```python
class CloudStorage:
    def __init__(self):
        self.files = {}       # file_id -> content
        self.permissions = {} # file_id -> set of user_ids

    def upload(self, file_id, content):
        self.files[file_id] = content

    def grant_access(self, file_id, user_id):
        self.permissions.setdefault(file_id, set()).add(user_id)

    def download(self, file_id, user_id):
        if user_id in self.permissions.get(file_id, set()):
            return self.files.get(file_id)
        return None
```

---

### 48. **设计一个图像识别服务**

**系统简介**
上传图像，返回识别结果（物体、文字等）。

**关键设计点**

* 图片上传与存储
* 模型推理接口
* 结果缓存与快速响应
* 多种识别模型支持

**核心类设计**

```python
class ImageRecognitionService:
    def __init__(self):
        self.model = self.load_model()
        self.cache = {}

    def load_model(self):
        # 加载预训练模型
        return "dummy_model"

    def recognize(self, image):
        if image in self.cache:
            return self.cache[image]
        result = f"Recognized objects in {image}"
        self.cache[image] = result
        return result
```

---

### 49. **设计一个电子商务库存管理系统**

**系统简介**
管理商品库存，订单扣减，补货提醒。

**关键设计点**

* 商品库存状态管理
* 订单事务处理
* 低库存告警
* 多仓库支持

**核心类设计**

```python
class InventorySystem:
    def __init__(self):
        self.stock = {}  # product_id -> quantity

    def add_stock(self, product_id, amount):
        self.stock[product_id] = self.stock.get(product_id, 0) + amount

    def reduce_stock(self, product_id, amount):
        if self.stock.get(product_id, 0) >= amount:
            self.stock[product_id] -= amount
            return True
        return False

    def check_low_stock(self, threshold=10):
        return [p for p, q in self.stock.items() if q < threshold]
```

---

### 50. **设计一个区块链基础系统**

**系统简介**
区块链账本，交易验证，链结构维护。

**关键设计点**

* 区块链结构（区块链，哈希链接）
* 挖矿与共识机制
* 交易池管理
* 节点同步

**核心类设计**

```python
import hashlib
import time

class Block:
    def __init__(self, index, previous_hash, timestamp, data, nonce=0):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = f"{self.index}{self.previous_hash}{self.timestamp}{self.data}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, "0", time.time(), "Genesis Block")

    def add_block(self, data):
        prev_block = self.chain[-1]
        new_block = Block(len(self.chain), prev_block.hash, time.time(), data)
        self.chain.append(new_block)
```

---

# 系统设计题目51-60

---

### 51. **设计一个在线问答社区（类似知乎）**

**系统简介**
用户提问、回答、点赞、关注。

**关键设计点**

* 问题、回答存储
* 用户关系管理（关注、点赞）
* 内容排序（热度、时间）
* 搜索功能

**核心类设计**

```python
class QASystem:
    def __init__(self):
        self.questions = {}  # question_id -> {'text':..., 'answers':[], 'likes':0}
        self.answers = {}    # answer_id -> {'text':..., 'likes':0}
        self.user_follows = {}  # user_id -> set of user_ids followed

    def post_question(self, question_id, text):
        self.questions[question_id] = {'text': text, 'answers': [], 'likes': 0}

    def post_answer(self, question_id, answer_id, text):
        self.answers[answer_id] = {'text': text, 'likes': 0}
        self.questions[question_id]['answers'].append(answer_id)

    def like_answer(self, answer_id):
        if answer_id in self.answers:
            self.answers[answer_id]['likes'] += 1

    def follow_user(self, user_id, target_user_id):
        self.user_follows.setdefault(user_id, set()).add(target_user_id)
```

---

### 52. **设计一个内容分发网络（CDN）**

**系统简介**
缓存静态内容，降低延迟。

**关键设计点**

* 边缘节点缓存
* 缓存失效策略（TTL、LRU）
* 请求路由到最近节点
* 源站回源机制

**核心类设计**

```python
class CDNNode:
    def __init__(self):
        self.cache = {}  # url -> (content, expiry_time)

    def get_content(self, url, current_time):
        if url in self.cache and self.cache[url][1] > current_time:
            return self.cache[url][0]
        else:
            content = self.fetch_from_origin(url)
            expiry = current_time + 3600  # 1 hour TTL
            self.cache[url] = (content, expiry)
            return content

    def fetch_from_origin(self, url):
        # 伪代码，向源服务器请求内容
        return f"Content for {url}"
```

---

### 53. **设计一个在线投票系统**

**系统简介**
支持多选投票、统计结果、防刷票。

**关键设计点**

* 投票唯一性验证
* 投票选项管理
* 实时结果统计
* 防止重复投票（IP限制、登录限制）

**核心类设计**

```python
class VotingSystem:
    def __init__(self):
        self.votes = {}      # vote_id -> {option -> count}
        self.user_votes = {} # user_id -> set of vote_ids

    def create_vote(self, vote_id, options):
        self.votes[vote_id] = {opt: 0 for opt in options}

    def submit_vote(self, user_id, vote_id, option):
        if vote_id in self.votes and option in self.votes[vote_id]:
            if user_id not in self.user_votes or vote_id not in self.user_votes[user_id]:
                self.votes[vote_id][option] += 1
                self.user_votes.setdefault(user_id, set()).add(vote_id)
                return True
        return False
```

---

### 54. **设计一个实时协作文档编辑系统**

**系统简介**
多人同时编辑文档，实时同步。

**关键设计点**

* 文档状态同步（基于操作转换OT或CRDT）
* 用户冲突处理
* 历史版本管理
* 用户权限控制

**核心类设计**

```python
class CollaborativeDoc:
    def __init__(self):
        self.content = ""
        self.operations = []  # 记录操作历史

    def apply_operation(self, op):
        # 简单追加操作示例，复杂应实现OT或CRDT算法
        self.content += op
        self.operations.append(op)

    def get_content(self):
        return self.content
```

---

### 55. **设计一个在线音乐流媒体平台**

**系统简介**
音乐上传、在线播放、推荐、播放列表。

**关键设计点**

* 音乐文件存储与转码
* 播放状态跟踪
* 用户歌单管理
* 推荐系统

**核心类设计**

```python
class MusicPlatform:
    def __init__(self):
        self.songs = {}      # song_id -> metadata
        self.playlists = {}  # user_id -> list of song_ids

    def upload_song(self, song_id, metadata):
        self.songs[song_id] = metadata

    def create_playlist(self, user_id):
        self.playlists[user_id] = []

    def add_to_playlist(self, user_id, song_id):
        if user_id in self.playlists and song_id in self.songs:
            self.playlists[user_id].append(song_id)
```

---

### 56. **设计一个智能家居控制系统**

**系统简介**
远程控制家电，自动化场景设置。

**关键设计点**

* 设备状态管理
* 事件触发和自动化规则
* 用户权限管理
* 实时通知

**核心类设计**

```python
class SmartHomeSystem:
    def __init__(self):
        self.devices = {}   # device_id -> status
        self.rules = []     # 自动化规则列表

    def update_device_status(self, device_id, status):
        self.devices[device_id] = status
        self.check_rules(device_id, status)

    def add_rule(self, rule):
        self.rules.append(rule)

    def check_rules(self, device_id, status):
        for rule in self.rules:
            if rule['device_id'] == device_id and rule['trigger'] == status:
                self.execute_action(rule['action'])

    def execute_action(self, action):
        print(f"执行动作: {action}")
```

---

### 57. **设计一个短信验证码服务**

**系统简介**
发送短信验证码，验证，防刷。

**关键设计点**

* 验证码生成与缓存
* 短信发送接口
* 限流和防刷策略
* 验证码过期

**核心类设计**

```python
import random
import time

class SMSService:
    def __init__(self):
        self.codes = {}  # phone -> (code, expiry_time)

    def send_code(self, phone):
        code = str(random.randint(100000, 999999))
        expiry = time.time() + 300  # 5分钟有效
        self.codes[phone] = (code, expiry)
        # 调用第三方短信接口发送 code
        print(f"发送短信验证码 {code} 到 {phone}")

    def verify_code(self, phone, code):
        if phone in self.codes:
            stored_code, expiry = self.codes[phone]
            if time.time() < expiry and stored_code == code:
                return True
        return False
```

---

### 58. **设计一个图片压缩服务**

**系统简介**
用户上传图片，压缩后返回。

**关键设计点**

* 支持多种压缩算法
* 任务异步处理
* 支持多种图片格式
* 文件存储和缓存

**核心类设计**

```python
from PIL import Image
import io

class ImageCompressor:
    def compress(self, image_bytes, quality=75):
        image = Image.open(io.BytesIO(image_bytes))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality)
        return output.getvalue()
```

---

### 59. **设计一个视频直播系统**

**系统简介**
主播推流，观众观看，弹幕互动。

**关键设计点**

* 实时视频转发和分发（RTMP、HLS）
* 弹幕实时推送
* 流量负载均衡
* 延迟控制

**核心类设计**

```python
class LiveStream:
    def __init__(self):
        self.streams = {}  # stream_id -> viewers list

    def start_stream(self, stream_id):
        self.streams[stream_id] = []

    def join_stream(self, stream_id, user_id):
        if stream_id in self.streams:
            self.streams[stream_id].append(user_id)

    def send_chat(self, stream_id, user_id, message):
        # 简单广播给所有观看者
        viewers = self.streams.get(stream_id, [])
        for viewer in viewers:
            print(f"{user_id} to {viewer}: {message}")
```

---

### 60. **设计一个广告投放系统**

**系统简介**
广告展示，竞价，统计效果。

**关键设计点**

* 广告竞价和排序
* 用户画像匹配
* 展示和点击统计
* 实时竞价处理

**核心类设计**

```python
class AdSystem:
    def __init__(self):
        self.ads = {}  # ad_id -> {bid, targeting}
        self.stats = {} # ad_id -> {impressions, clicks}

    def add_ad(self, ad_id, bid, targeting):
        self.ads[ad_id] = {'bid': bid, 'targeting': targeting}
        self.stats[ad_id] = {'impressions': 0, 'clicks': 0}

    def match_ads(self, user_profile):
        candidates = []
        for ad_id, ad in self.ads.items():
            if self.match_targeting(ad['targeting'], user_profile):
                candidates.append((ad_id, ad['bid']))
        candidates.sort(key=lambda x: -x[1])  # 竞价排序
        return candidates[0][0] if candidates else None

    def match_targeting(self, targeting, profile):
        # 简单匹配示例
        return all(item in profile.items() for item in targeting.items())
```

---

# 系统设计题目61-70

---

### 61. **设计一个任务调度系统（类似Cron）**

**系统简介**
定时执行任务，支持重复任务、失败重试。

**关键设计点**

* 任务队列
* 定时触发器
* 任务状态管理
* 失败重试机制

**核心类设计**

```python
import threading
import time
import heapq

class ScheduledTask:
    def __init__(self, run_at, func, interval=None):
        self.run_at = run_at
        self.func = func
        self.interval = interval  # 重复间隔，单位秒

    def __lt__(self, other):
        return self.run_at < other.run_at

class Scheduler:
    def __init__(self):
        self.tasks = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def schedule(self, task):
        with self.lock:
            heapq.heappush(self.tasks, task)

    def run(self):
        while True:
            with self.lock:
                if self.tasks and self.tasks[0].run_at <= time.time():
                    task = heapq.heappop(self.tasks)
                    try:
                        task.func()
                    except Exception as e:
                        print(f"任务执行失败: {e}")
                    if task.interval:
                        task.run_at = time.time() + task.interval
                        heapq.heappush(self.tasks, task)
                else:
                    pass
            time.sleep(0.5)
```

---

### 62. **设计一个分布式锁系统**

**系统简介**
保证多实例环境下资源互斥访问。

**关键设计点**

* 锁申请和释放
* 超时自动释放
* 高可用（可用Redis或Zookeeper实现）

**核心类设计（简化Redis模拟）**

```python
import time
import threading

class DistributedLock:
    def __init__(self):
        self.locks = {}
        self.lock = threading.Lock()

    def acquire(self, key, ttl=10):
        now = time.time()
        with self.lock:
            if key not in self.locks or self.locks[key] < now:
                self.locks[key] = now + ttl
                return True
            else:
                return False

    def release(self, key):
        with self.lock:
            if key in self.locks:
                del self.locks[key]
```

---

### 63. **设计一个在线教育平台**

**系统简介**
课程发布、视频播放、作业提交。

**关键设计点**

* 课程管理
* 视频存储和播放
* 用户进度跟踪
* 交互和作业提交

**核心类设计**

```python
class Course:
    def __init__(self, course_id, title):
        self.course_id = course_id
        self.title = title
        self.videos = []
        self.assignments = []

class EducationPlatform:
    def __init__(self):
        self.courses = {}
        self.user_progress = {}  # user_id -> {course_id: progress}

    def add_course(self, course_id, title):
        self.courses[course_id] = Course(course_id, title)

    def add_video(self, course_id, video_url):
        if course_id in self.courses:
            self.courses[course_id].videos.append(video_url)

    def update_progress(self, user_id, course_id, progress):
        self.user_progress.setdefault(user_id, {})[course_id] = progress
```

---

### 64. **设计一个电商推荐系统**

**系统简介**
基于用户行为推荐商品。

**关键设计点**

* 用户行为采集
* 商品相似度计算
* 实时推荐
* 冷启动问题

**核心类设计**

```python
class RecommenderSystem:
    def __init__(self):
        self.user_history = {}  # user_id -> set of item_ids
        self.item_similarities = {}  # item_id -> list of (similar_item_id, score)

    def record_view(self, user_id, item_id):
        self.user_history.setdefault(user_id, set()).add(item_id)

    def recommend(self, user_id, top_n=5):
        viewed = self.user_history.get(user_id, set())
        scores = {}
        for item in viewed:
            for sim_item, score in self.item_similarities.get(item, []):
                if sim_item not in viewed:
                    scores[sim_item] = scores.get(sim_item, 0) + score
        return sorted(scores.items(), key=lambda x: -x[1])[:top_n]
```

---

### 65. **设计一个支付网关系统**

**系统简介**
处理多种支付方式，订单管理。

**关键设计点**

* 支付请求处理
* 订单状态跟踪
* 第三方支付集成
* 安全与幂等

**核心类设计**

```python
class PaymentGateway:
    def __init__(self):
        self.orders = {}  # order_id -> status

    def create_order(self, order_id, amount):
        self.orders[order_id] = 'pending'

    def process_payment(self, order_id, payment_method):
        # 调用第三方接口，伪代码
        success = True
        if success:
            self.orders[order_id] = 'paid'
        else:
            self.orders[order_id] = 'failed'

    def get_status(self, order_id):
        return self.orders.get(order_id, 'unknown')
```

---

### 66. **设计一个文件同步系统（类似Dropbox）**

**系统简介**
多设备间文件同步，冲突解决。

**关键设计点**

* 文件版本管理
* 增量同步
* 冲突检测与合并
* 离线支持

**核心类设计**

```python
class FileSyncSystem:
    def __init__(self):
        self.files = {}  # filename -> {version, content}

    def upload_file(self, filename, version, content):
        if filename not in self.files or version > self.files[filename]['version']:
            self.files[filename] = {'version': version, 'content': content}
            return True
        return False

    def get_file(self, filename):
        return self.files.get(filename)
```

---

### 67. **设计一个短链接服务**

**系统简介**
长链接转短链接，统计访问。

**关键设计点**

* 短链接生成
* 长链接映射
* 访问统计
* 防止碰撞

**核心类设计**

```python
import hashlib

class URLShortener:
    def __init__(self):
        self.mapping = {}  # short -> long

    def shorten(self, long_url):
        short_key = hashlib.md5(long_url.encode()).hexdigest()[:6]
        self.mapping[short_key] = long_url
        return short_key

    def retrieve(self, short_key):
        return self.mapping.get(short_key)
```

---

### 68. **设计一个在线书籍阅读平台**

**系统简介**
电子书上传、章节阅读、书签。

**关键设计点**

* 书籍管理
* 阅读进度跟踪
* 书签功能
* 支持多格式

**核心类设计**

```python
class Book:
    def __init__(self, book_id, chapters):
        self.book_id = book_id
        self.chapters = chapters

class ReadingPlatform:
    def __init__(self):
        self.books = {}
        self.user_progress = {}  # user_id -> {book_id: chapter}

    def add_book(self, book_id, chapters):
        self.books[book_id] = Book(book_id, chapters)

    def update_progress(self, user_id, book_id, chapter):
        self.user_progress.setdefault(user_id, {})[book_id] = chapter
```

---

### 69. **设计一个订单配送跟踪系统**

**系统简介**
跟踪订单配送状态，通知用户。

**关键设计点**

* 订单状态管理
* 位置更新接口
* 用户通知
* 历史轨迹保存

**核心类设计**

```python
class DeliverySystem:
    def __init__(self):
        self.orders = {}  # order_id -> status
        self.locations = {}  # order_id -> (lat, lng)

    def update_status(self, order_id, status):
        self.orders[order_id] = status

    def update_location(self, order_id, lat, lng):
        self.locations[order_id] = (lat, lng)

    def get_order_info(self, order_id):
        return self.orders.get(order_id), self.locations.get(order_id)
```

---

### 70. **设计一个社交媒体点赞和评论系统**

**系统简介**
支持点赞、评论、回复。

**关键设计点**

* 点赞计数和状态
* 评论结构（树形）
* 用户通知
* 幂等操作

**核心类设计**

```python
class SocialPost:
    def __init__(self):
        self.likes = set()  # user_ids
        self.comments = {}  # comment_id -> {'user_id', 'text', 'replies':[]}

    def like(self, user_id):
        self.likes.add(user_id)

    def unlike(self, user_id):
        self.likes.discard(user_id)

    def add_comment(self, comment_id, user_id, text, parent_comment_id=None):
        self.comments[comment_id] = {'user_id': user_id, 'text': text, 'replies': []}
        if parent_comment_id:
            self.comments[parent_comment_id]['replies'].append(comment_id)
```

---

# 系统设计题目71-80

---

### 71. **设计一个视频直播系统**

**系统简介**
支持多用户实时观看，低延迟推流。

**关键设计点**

* 流媒体服务器（如使用RTMP/HTTP-FLV）
* CDN分发加速
* 观众管理和互动（弹幕、点赞）
* 录制与回放

**核心类设计**

```python
class LiveStream:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.viewers = set()

    def add_viewer(self, user_id):
        self.viewers.add(user_id)

    def remove_viewer(self, user_id):
        self.viewers.discard(user_id)

    def get_viewer_count(self):
        return len(self.viewers)
```

---

### 72. **设计一个图片识别API服务**

**系统简介**
客户端上传图片，返回识别结果。

**关键设计点**

* 图片上传接口
* 异步任务队列处理
* AI模型推理调用
* 结果缓存和快速响应

**核心类设计**

```python
import queue
import threading

class ImageRecognitionService:
    def __init__(self):
        self.task_queue = queue.Queue()
        self.results = {}
        threading.Thread(target=self.worker, daemon=True).start()

    def submit_task(self, image_id, image_data):
        self.task_queue.put((image_id, image_data))

    def worker(self):
        while True:
            image_id, image_data = self.task_queue.get()
            # 调用AI模型识别，伪代码
            result = self.recognize(image_data)
            self.results[image_id] = result
            self.task_queue.task_done()

    def recognize(self, image_data):
        # 模拟AI识别过程
        return {"labels": ["cat", "animal"]}

    def get_result(self, image_id):
        return self.results.get(image_id)
```

---

### 73. **设计一个新闻推荐系统**

**系统简介**
根据用户兴趣和热点推荐新闻。

**关键设计点**

* 新闻内容分类和标签
* 用户兴趣建模
* 热点新闻识别
* 实时推荐

**核心类设计**

```python
class NewsRecommender:
    def __init__(self):
        self.user_profiles = {}  # user_id -> set of interests
        self.news = {}  # news_id -> {'title', 'tags'}

    def add_news(self, news_id, title, tags):
        self.news[news_id] = {'title': title, 'tags': set(tags)}

    def update_user_profile(self, user_id, interests):
        self.user_profiles[user_id] = set(interests)

    def recommend(self, user_id, top_n=5):
        interests = self.user_profiles.get(user_id, set())
        scored = []
        for nid, info in self.news.items():
            score = len(interests.intersection(info['tags']))
            if score > 0:
                scored.append((score, nid))
        scored.sort(reverse=True)
        return [nid for _, nid in scored[:top_n]]
```

---

### 74. **设计一个文档协作编辑系统**

**系统简介**
多用户同时编辑同一文档，实时同步。

**关键设计点**

* 实时同步与冲突解决（OT或CRDT算法）
* 用户权限管理
* 历史版本控制

**核心类设计**

```python
class Document:
    def __init__(self, doc_id, content=""):
        self.doc_id = doc_id
        self.content = content
        self.version = 0

    def update(self, new_content, version):
        if version == self.version:
            self.content = new_content
            self.version += 1
            return True
        else:
            # 版本冲突，需要合并
            return False
```

---

### 75. **设计一个在线投票系统**

**系统简介**
支持用户投票，防止重复投票。

**关键设计点**

* 投票权限验证
* 投票计数和实时统计
* 防止作弊（IP限制、验证码）

**核心类设计**

```python
class VotingSystem:
    def __init__(self):
        self.votes = {}  # option -> count
        self.voters = set()

    def vote(self, user_id, option):
        if user_id in self.voters:
            return False  # 已投票
        self.votes[option] = self.votes.get(option, 0) + 1
        self.voters.add(user_id)
        return True

    def get_results(self):
        return self.votes
```

---

### 76. **设计一个日志收集与分析系统**

**系统简介**
收集应用日志，支持查询和统计。

**关键设计点**

* 日志采集代理
* 日志存储（分布式，按时间分片）
* 实时和离线分析

**核心类设计**

```python
class LogCollector:
    def __init__(self):
        self.logs = []

    def collect(self, log_entry):
        self.logs.append(log_entry)

    def query(self, keyword):
        return [log for log in self.logs if keyword in log]
```

---

### 77. **设计一个聊天机器人服务**

**系统简介**
用户输入问题，机器人智能回复。

**关键设计点**

* 自然语言理解（NLP）
* 对话管理
* 多轮对话上下文维护

**核心类设计**

```python
class ChatBot:
    def __init__(self):
        self.context = {}

    def ask(self, user_id, question):
        # 简化逻辑：返回固定回答
        return "这是机器人回答。"
```

---

### 78. **设计一个内容分发网络（CDN）**

**系统简介**
缓存静态资源，提高访问速度。

**关键设计点**

* 节点选择
* 缓存策略（LRU, TTL）
* 缓存更新和失效

**核心类设计**

```python
class CDNCache:
    def __init__(self, capacity=100):
        self.cache = {}
        self.capacity = capacity
        self.order = []

    def get(self, url):
        if url in self.cache:
            self.order.remove(url)
            self.order.append(url)
            return self.cache[url]
        return None

    def put(self, url, content):
        if url in self.cache:
            self.order.remove(url)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[url] = content
        self.order.append(url)
```

---

### 79. **设计一个多人游戏匹配系统**

**系统简介**
玩家匹配进入游戏房间。

**关键设计点**

* 匹配算法（基于等级、延迟等）
* 房间管理
* 并发控制

**核心类设计**

```python
class MatchMaking:
    def __init__(self):
        self.waiting_players = []

    def join_queue(self, player):
        self.waiting_players.append(player)

    def match_players(self):
        matches = []
        while len(self.waiting_players) >= 2:
            p1 = self.waiting_players.pop(0)
            p2 = self.waiting_players.pop(0)
            matches.append((p1, p2))
        return matches
```

---

### 80. **设计一个文件上传服务**

**系统简介**
支持大文件分片上传，断点续传。

**关键设计点**

* 分片存储
* 分片校验和合并
* 上传状态管理

**核心类设计**

```python
class FileUploadService:
    def __init__(self):
        self.uploads = {}  # upload_id -> {chunk_index: data}

    def upload_chunk(self, upload_id, chunk_index, data):
        if upload_id not in self.uploads:
            self.uploads[upload_id] = {}
        self.uploads[upload_id][chunk_index] = data

    def is_complete(self, upload_id, total_chunks):
        return len(self.uploads.get(upload_id, {})) == total_chunks

    def merge_chunks(self, upload_id, total_chunks):
        chunks = self.uploads[upload_id]
        file_data = b''.join(chunks[i] for i in range(total_chunks))
        return file_data
```

# 系统设计题目81-90

---

### 81. **分布式任务队列系统**  
**系统简介**  
异步处理耗时任务，支持任务分发、状态跟踪和失败重试。  

**关键设计点**  
* 生产者-消费者模型  
* 任务优先级和延迟执行  
* 失败重试和死信队列  
* 水平扩展和负载均衡  

**核心类设计**  
```python
import heapq
import threading
import time

class TaskQueue:
    def __init__(self):
        self.queue = []
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.worker_pool = []
        
    def add_task(self, task, priority=0, delay=0):
        with self.lock:
            execute_at = time.time() + delay
            heapq.heappush(self.queue, (priority, execute_at, task))
            self.cond.notify()
    
    def start_worker(self, num_workers=3):
        for _ in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.worker_pool.append(worker)
    
    def _worker_loop(self):
        while True:
            with self.lock:
                while not self.queue or self.queue[0][1] > time.time():
                    if not self.queue:
                        self.cond.wait()
                    else:
                        timeout = self.queue[0][1] - time.time()
                        self.cond.wait(timeout=max(0, timeout))
                
                _, _, task = heapq.heappop(self.queue)
            
            try:
                task.execute()
            except Exception as e:
                print(f"任务失败: {e}")
                task.retry_count += 1
                if task.retry_count < 3:
                    self.add_task(task, delay=60)  # 1分钟后重试

class Task:
    def __init__(self, name):
        self.name = name
        self.retry_count = 0
    
    def execute(self):
        print(f"执行任务: {self.name}")
        # 真实任务逻辑
```

---

### 82. **实时交通监控系统**  
**系统简介**  
收集车辆位置数据，实时计算道路拥堵情况并提供路线规划。  

**关键设计点**  
* 海量GPS数据处理  
* 实时路况计算（滑动窗口）  
* 路线规划算法优化  
* 地理围栏技术  

**核心类设计**  
```python
from collections import deque
import numpy as np

class TrafficMonitor:
    def __init__(self, grid_size=0.01, window_size=60):
        # 0.01度≈1.1km网格
        self.grid_traffic = {}
        self.window_size = window_size
        self.grid_size = grid_size
    
    def _get_grid_id(self, lat, lon):
        return (int(lat / self.grid_size), int(lon / self.grid_size))
    
    def update_vehicle(self, vehicle_id, lat, lon, timestamp):
        grid_id = self._get_grid_id(lat, lon)
        
        if grid_id not in self.grid_traffic:
            self.grid_traffic[grid_id] = deque(maxlen=self.window_size)
        
        self.grid_traffic[grid_id].append(timestamp)
    
    def get_congestion_level(self, grid_id):
        if grid_id not in self.grid_traffic:
            return 0
        
        timestamps = self.grid_traffic[grid_id]
        if len(timestamps) < 2:
            return 0
        
        # 计算平均时间间隔
        intervals = np.diff(sorted(timestamps))
        avg_interval = np.mean(intervals)
        
        # 间隔越小表示越拥堵
        return min(1.0, max(0, 1 - avg_interval/60)) 
```

---

### 83. **智能家居能源管理系统**  
**系统简介**  
监控家电能耗，优化用电策略，降低能源成本。  

**关键设计点**  
* 设备状态实时监控  
* 电价时段预测  
* 负载均衡调度  
* 异常用电检测  

**核心类设计**  
```python
import datetime

class EnergyManager:
    def __init__(self):
        self.devices = {}  # device_id: {power, schedule}
        self.price_schedule = {
            (0, 7): 0.08,    # 谷电
            (7, 19): 0.15,   # 峰电
            (19, 24): 0.12   # 平电
        }
    
    def add_device(self, device_id, power, schedule=None):
        self.devices[device_id] = {
            'power': power,         # 功率 (kW)
            'schedule': schedule,   # 运行时段
            'status': 'off'
        }
    
    def optimize_schedule(self):
        current_hour = datetime.datetime.now().hour
        current_price = next(
            price for (start, end), price in self.price_schedule.items() 
            if start <= current_hour < end
        )
        
        for device_id, data in self.devices.items():
            if data['schedule'] is None:
                # 自动调度高能耗设备到低价时段
                if data['power'] > 1.5 and current_price > 0.1:
                    data['status'] = 'delayed'
                else:
                    data['status'] = 'on'
    
    def detect_anomaly(self):
        total_power = sum(
            data['power'] for data in self.devices.values() 
            if data['status'] == 'on'
        )
        return total_power > 5.0  # 异常阈值
```

---

### 84. **医疗影像分析系统**  
**系统简介**  
处理DICOM影像，运行AI模型检测病变，生成诊断报告。  

**关键设计点**  
* DICOM数据处理  
* GPU加速推理  
* 敏感数据加密  
* 报告版本管理  

**核心类设计**  
```python
import hashlib
import numpy as np

class MedicalImagingSystem:
    def __init__(self, model):
        self.model = model
        self.storage = {}
        self.encryption_key = b'secure_key_123'
    
    def _encrypt(self, data):
        # AES加密简化示例
        return hashlib.sha256(data + self.encryption_key).digest()
    
    def upload_dicom(self, patient_id, dicom_data):
        encrypted = self._encrypt(dicom_data)
        self.storage[patient_id] = encrypted
    
    def analyze_image(self, patient_id):
        encrypted = self.storage[patient_id]
        dicom_data = self._decrypt(encrypted)
        
        # 转换为模型输入格式
        image_array = self._dicom_to_array(dicom_data)
        result = self.model.predict(image_array)
        
        return {
            'findings': result['abnormalities'],
            'confidence': result['confidence']
        }
    
    def generate_report(self, analysis):
        return f"""医疗报告：
        发现异常: {analysis['findings']}
        置信度: {analysis['confidence']*100:.2f}%"""
```

---

### 85. **区块链智能合约平台**  
**系统简介**  
执行去中心化智能合约，确保不可篡改的交易记录。  

**关键设计点**  
* 共识机制（PoW/PoS）  
* 智能合约沙箱执行  
* 交易验证  
* 分片存储  

**核心类设计**  
```python
import hashlib
import json
from datetime import datetime

class Blockchain:
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.create_block(proof=1, previous_hash='0')
    
    def create_block(self, proof, previous_hash):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': str(datetime.now()),
            'transactions': self.pending_transactions,
            'proof': proof,
            'previous_hash': previous_hash
        }
        self.pending_transactions = []
        self.chain.append(block)
        return block
    
    def add_transaction(self, sender, receiver, amount, contract=None):
        transaction = {
            'sender': sender,
            'receiver': receiver,
            'amount': amount,
            'contract': contract
        }
        self.pending_transactions.append(transaction)
        return self.last_block['index'] + 1
    
    @property
    def last_block(self):
        return self.chain[-1]
    
    def proof_of_work(self, last_proof):
        proof = 0
        while not self.valid_proof(last_proof, proof):
            proof += 1
        return proof
    
    @staticmethod
    def valid_proof(last_proof, proof):
        guess = f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
```

---

### 86. **多语言实时翻译系统**  
**系统简介**  
支持100+语言实时互译，保留上下文语义。  

**关键设计点**  
* 神经机器翻译模型  
* 上下文记忆管理  
* 低延迟流式处理  
* 领域自适应  

**核心类设计**  
```python
class TranslationSession:
    def __init__(self, user_id, src_lang, tgt_lang):
        self.user_id = user_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.context_memory = []
        self.model = self.load_model()
    
    def load_model(self):
        # 加载预训练模型
        return "transformer_model"
    
    def translate(self, text):
        # 添加上下文
        full_input = " ".join(self.context_memory[-3:] + [text])
        
        # 执行翻译
        translated = self.model.translate(
            full_input, 
            src_lang=self.src_lang, 
            tgt_lang=self.tgt_lang
        )
        
        # 更新上下文
        self.context_memory.append(text)
        return translated
    
    def stream_translate(self, text_stream):
        buffer = ""
        for chunk in text_stream:
            buffer += chunk
            if any(punct in buffer for punct in ['.', '?', '!', '。', '？', '！']):
                # 遇到句子边界时翻译
                yield self.translate(buffer)
                buffer = ""
        
        if buffer:
            yield self.translate(buffer)
```

---

### 87. **3D游戏物理引擎**  
**系统简介**  
模拟真实世界物理效果：碰撞检测、刚体运动、流体动力学。  

**关键设计点**  
* 连续碰撞检测(CCD)  
* 约束求解器  
* 空间分割优化  
* 多线程计算  

**核心类设计**  
```python
import numpy as np
from scipy.spatial import KDTree

class PhysicsEngine:
    def __init__(self, gravity=(0, -9.8, 0)):
        self.objects = []
        self.gravity = np.array(gravity)
        self.collision_pairs = set()
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def update(self, dt):
        # 运动积分
        for obj in self.objects:
            if obj.mass > 0:
                obj.velocity += (self.gravity + obj.force / obj.mass) * dt
                obj.position += obj.velocity * dt
        
        # 碰撞检测
        self._detect_collisions()
        
        # 解决碰撞
        self._resolve_collisions()
    
    def _detect_collisions(self):
        positions = [obj.position for obj in self.objects]
        tree = KDTree(positions)
        
        self.collision_pairs = set()
        for i, obj1 in enumerate(self.objects):
            neighbors = tree.query_ball_point(positions[i], obj1.radius * 2)
            for j in neighbors:
                if i != j and self._check_collision(obj1, self.objects[j]):
                    self.collision_pairs.add((min(i, j), max(i, j)))
    
    def _check_collision(self, obj1, obj2):
        distance = np.linalg.norm(obj1.position - obj2.position)
        return distance < (obj1.radius + obj2.radius)
    
    def _resolve_collisions(self):
        for i, j in self.collision_pairs:
            obj1, obj2 = self.objects[i], self.objects[j]
            # 动量守恒碰撞解决
            # ... (具体实现省略)
```

---

### 88. **金融风险实时计算引擎**  
**系统简介**  
监控市场数据，实时计算投资组合风险指标。  

**关键设计点**  
* 流式数据处理  
* VaR(风险价值)计算  
* 压力测试场景  
* 实时预警系统  

**核心类设计**  
```python
import numpy as np
from scipy.stats import norm

class RiskEngine:
    def __init__(self, confidence_level=0.95, window=252):
        self.portfolio = {}
        self.historical_returns = {}
        self.confidence_level = confidence_level
        self.window = window  # 交易日窗口
    
    def add_position(self, asset_id, quantity):
        self.portfolio[asset_id] = quantity
    
    def update_price(self, asset_id, price):
        if asset_id not in self.historical_returns:
            self.historical_returns[asset_id] = []
        
        if len(self.historical_returns[asset_id]) > 0:
            last_price = self.historical_returns[asset_id][-1]['price']
            returns = (price - last_price) / last_price
            self.historical_returns[asset_id].append({
                'price': price,
                'return': returns
            })
            # 保持窗口大小
            if len(self.historical_returns[asset_id]) > self.window:
                self.historical_returns[asset_id].pop(0)
    
    def calculate_var(self):
        portfolio_value = sum(
            self.historical_returns[asset_id][-1]['price'] * quantity
            for asset_id, quantity in self.portfolio.items()
        )
        
        # 计算组合收益率
        portfolio_returns = []
        for i in range(1, self.window):
            daily_return = 0
            for asset_id, quantity in self.portfolio.items():
                if len(self.historical_returns[asset_id]) > i:
                    asset_return = self.historical_returns[asset_id][-i]['return']
                    weight = self.historical_returns[asset_id][-i]['price'] * quantity / portfolio_value
                    daily_return += weight * asset_return
            portfolio_returns.append(daily_return)
        
        # 计算VaR
        mean = np.mean(portfolio_returns)
        std = np.std(portfolio_returns)
        z_score = norm.ppf(1 - self.confidence_level)
        var = portfolio_value * (mean - z_score * std)
        
        return max(0, var)
```

---

### 89. **无人机集群控制系统**  
**系统简介**  
协调多无人机执行协同任务：编队飞行、区域搜索。  

**关键设计点**  
* 分布式共识算法  
* 避障路径规划  
* 通信延迟补偿  
* 故障转移机制  

**核心类设计**  
```python
import numpy as np
from scipy.spatial.distance import cdist

class DroneSwarm:
    def __init__(self, num_drones):
        self.drones = [Drone(i) for i in range(num_drones)]
        self.formation = self._default_formation()
    
    def _default_formation(self):
        # 三角形编队
        return np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.87, 0]
        ])
    
    def update_positions(self):
        positions = np.array([drone.position for drone in self.drones])
        
        # 计算编队目标位置
        centroid = np.mean(positions, axis=0)
        target_positions = centroid + self.formation
        
        # 分配目标位置（匈牙利算法）
        cost_matrix = cdist(positions, target_positions)
        assignments = self._hungarian_algorithm(cost_matrix)
        
        # 更新速度
        for i, j in enumerate(assignments):
            direction = target_positions[j] - positions[i]
            distance = np.linalg.norm(direction)
            if distance > 0.1:
                self.drones[i].velocity = direction / distance * min(2, distance)
    
    def avoid_obstacles(self, obstacles):
        for drone in self.drones:
            for obstacle in obstacles:
                # 计算避障力
                # ... (具体实现省略)
                pass
    
    def _hungarian_algorithm(self, cost_matrix):
        # 匈牙利算法实现
        # ... (具体实现省略)
        return [i for i in range(cost_matrix.shape[0])]

class Drone:
    def __init__(self, id):
        self.id = id
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.battery = 100
```

---

### 90. **量子计算模拟框架**  
**系统简介**  
模拟量子计算机行为，运行量子算法。  

**关键设计点**  
* 量子态表示（状态向量）  
* 量子门操作  
* 噪声模型模拟  
* 并行化计算  

**核心类设计**  
```python
import numpy as np
from scipy.sparse import kron, eye

class QuantumSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1  # 初始状态 |0...0>
        
        # 基础量子门
        self.X = np.array([[0, 1], [1, 0]])
        self.H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self.CNOT = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
    
    def apply_gate(self, gate, target_qubits):
        # 构建完整变换矩阵
        full_gate = self._build_gate(gate, target_qubits)
        self.state = full_gate.dot(self.state)
    
    def _build_gate(self, gate, target_qubits):
        # 创建单位矩阵张量积
        gate_size = gate.shape[0]
        total_size = 2**self.num_qubits
        full_gate = np.eye(total_size, dtype=complex)
        
        # 对每个目标量子位应用门
        for qubit in target_qubits:
            # 创建作用于单个量子位的门
            ops = [np.eye(2)] * self.num_qubits
            ops[qubit] = gate
            
            # 构建完整变换
            current_gate = ops[0]
            for op in ops[1:]:
                current_gate = np.kron(current_gate, op)
            
            full_gate = current_gate.dot(full_gate)
        
        return full_gate
    
    def measure(self, qubit):
        # 计算测量概率
        prob_0 = 0
        for i in range(len(self.state)):
            if i & (1 << qubit) == 0:  # 检查目标量子位是否为0
                prob_0 += np.abs(self.state[i])**2
        
        # 随机测量结果
        result = 0 if np.random.random() < prob_0 else 1
        
        # 坍缩量子态
        # ... (具体实现省略)
        
        return result
```


