'''
设计一个日志记录系统，使得每条消息都只能每 10 秒 打印一次。
timestamp 是当前时间（单位是秒）
如果消息在过去 10 秒内 已经被打印过，则不能重复打印
否则，返回 true，并认为该消息被打印了

logger = Logger()
logger.shouldPrintMessage(1, "foo")      # returns True
logger.shouldPrintMessage(2, "bar")      # returns True
logger.shouldPrintMessage(3, "foo")      # returns False
logger.shouldPrintMessage(8, "bar")      # returns False
logger.shouldPrintMessage(10, "foo")     # returns False
logger.shouldPrintMessage(11, "foo")     # returns True
'''

class Logger:
    def __init__(self):
        self.d = {}

    def shouldPrintMessage(self, timestamp, message) :
        if message in self.d:
            if timestamp - self.d[message] < 10:
                return False
            else:
                self.d[message] = timestamp
                return True
        else:
            self.d[message] = timestamp
            return True

# HashMap 会变大，怎么办？
'''
1. 在单机版中我们用 dict 存储 message → timestamp
2. 如果系统需要支撑百万级吞吐，我会采用 Redis 作为限流状态中心，设置自动过期的 Key。
3. 为了进一步优化系统空间占用和吞吐能力，我会引入 滑动窗口（Sliding Window）或时间轮（Time Wheel）结构，用于管理消息打印的时效性。
这两种机制都可以避免传统 HashMap 可能带来的内存泄漏问题，同时将过期消息批量清理为 O(1) 复杂度，提高系统整体运行效率。

时间轮结构将时间轴切分为固定窗口（例如10秒一个轮次），并将待清理的 message 映射至未来对应时间槽中。
每秒钟只需清理当前槽位内的消息即可，无需全量遍历，大幅降低 GC 压力和 CPU 使用率。

滑动窗口计数器则适合处理更复杂的限流场景（如 10 秒内最多打印 N 次），可以对用户/消息组合做精细化控制。

4. 此外，为了解耦日志的生产与消费，提升系统可扩展性与异步处理能力，我会引入 消息队列（如 Kafka 或 Pulsar）。
日志消息会首先被写入队列，由专门的日志消费服务异步进行限流判断与落库操作，
“我们通过 Redis 控制消息是否允许打印，如果允许，则异步将日志写入 Elasticsearch（用于实时查询）和 HDFS（用于归档与大数据分析）。
这种架构既保证了在线系统的性能，又支持后期的数据可用性。”


“TikTok 这类系统本质是异步日志风控+行为限流系统，可以参考 RateLimiter + Redis + TimeWheel 架构设计。”
| 方式            | 特点                  | 是否推荐用于分布式  |
| ------------- | ------------------- | ---------- |
| HashMap + TTL | 简单快速，适用于单机          | ❌          |
| Redis + TTL   | 分布式强一致性，自动清理        | ✅ 强烈推荐     |
| 时间轮算法         | O(1) 定时器清理，适合大量定时任务 | ✅（需加持状态同步） |


[ User Request ]
       ↓
 ┌────────────┐
 │ API Gateway│
 └────┬───────┘
      ↓
┌──────────────┐
│ Kafka/Queue  │ ← message 封装后写入队列
└────┬─────────┘
     ↓
┌────────────────────────────┐
│ Async Logger Worker Pool   │
│ - Redis TTL 判断            │
│ - 时间轮 / 滑动窗口更新     │
│ - 允许时写日志（ES/HDFS）  │
└────────────────────────────┘

'''

class Logger2:
    def __init__(self):
        self.d = {}

    def shouldPrintMessage(self, timestamp, message) :
        '''
        1.Redis TTL 判断
        2. 时间轮
        3. 滑动窗口更新
        '''
        pass

logger = Logger()

print(logger.shouldPrintMessage(1, "foo"))      # returns True
print(logger.shouldPrintMessage(2, "bar"))      # returns True
print(logger.shouldPrintMessage(3, "foo"))     # returns False
print(logger.shouldPrintMessage(8, "bar"))      # returns False
print(logger.shouldPrintMessage(10, "foo"))     # returns False
print(logger.shouldPrintMessage(11, "foo"))     # returns True