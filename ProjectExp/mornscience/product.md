# MornScience Product — Agent CTO 产品与技术总览

> 用途：简历「Agent CTO / 技术一号位」叙事底稿 + 面试白板材料  
> 口径：**15 款产品完成 0→1**（落入 **5 大热门 Agent 面试赛道**）+ **3 款 TBD**  
> 数据源：`市场需求挖掘1.xlsx` · GitHub `yuxuanzhouo3/mvp_1…mvp_35` · `yzcmf/mornscience` · 简历详细版 → v2

---

## 1. 一句话定位（简历可用）

从 0 到 1 创建 **MornScience**，担任 **CEO / 技术总负责人（Agent CTO）**：基于市场需求挖掘体系，主导 **15 款 AI Agent 产品完成 0→1**，覆盖 **5 大热门 Agent 面试赛道**（多智能体编排 · Coding/端生成 · 垂直业务 Agent · 匹配/双边市场 · 工具自动化/多模态）；另有 **3 款 TBD** 在商用完善中。代码资产沉淀于 GitHub **mvp_1–mvp_35**，并建立 **Agent Eval / Architecture / Harness** 三件套，支撑「内部 AI Native → PMF 验证 → 产品商用」闭环。

---

## 2. 信息源与代码资产

| 类型 | 位置 |
|------|------|
| 市场需求与 CTO 面试表 | `/Users/mac-guest1/Downloads/晨佑科学创业/市场/市场需求挖掘1.xlsx`（含 Agent 全栈组件列 + CTO 三阶段设问列） |
| 产品代码（MVP 1–35） | https://github.com/yuxuanzhouo3 （`mvp_1` … `mvp_35`） |
| 公司产品仓 | https://github.com/yzcmf/mornscience |
| Agent 评测 Harness | https://github.com/yuxuanzhouo3/harbor （Harbor：评估与优化 Agent / LM） |
| 简历底稿 | `周宇轩简历(详细版).docx` → 下一代见同目录 `周宇轩简历(详细版_AgentCTO_v2).*` |

**全栈组件统一检查项（Excel L 列）**：登录 · 支付 · 核心功能 · AI Agent · Admin · Market · 客户端（iOS / Android / 微信小程序 / Web / Chrome 扩展 / Mac / Win / Linux）

**CTO 面试三阶段（Excel N 列）**：内部 AI Native → PMF 验证 → 产品商用

---

## 3. 简历口径：15（0→1）落入 5 大热门 Agent 面试赛道 + 3 TBD

### 3.1 五大赛道 × 十五款已完成 0→1

| 赛道（面试高频） | 产品（各 3 款） | 面试要打穿的能力点 |
|------------------|-----------------|-------------------|
| **A. 多智能体编排 / Agent 平台** | **multigpt** · **morngpt** · **mornbusiness** | Supervisor–Worker、专家路由、商业 OS 任务图、席位/工作流售卖 |
| **B. Coding Agent / 端与工程生成** | **mornclient** · **mornfront** · **mornsite** | NL/URL→多端产物、前端工程资产、扩展增长表面与构建熔断 |
| **C. 垂直业务 Agent** | **morncontract** · **morncoach** · **mornspeaker** | 合同结构化校验与免责、提分/Offer 结果售卖、实时多语延迟预算 |
| **D. 匹配 / 双边市场 Agent** | **mornsocial** · **mornxyz** · **mornhome** | 排序+解释、意图澄清深链、双边撮合与押金信任状态机 |
| **E. 工具自动化 / 多模态 Agent** | **morntool** · **mornfake** · **mornchat** | OS 动作沙箱、生成+鉴伪双引擎、IM 旁路 Agent 不拖垮主链路 |

**产品细表**

| 赛道 | 产品 | MVP | 需求/商用 | Agent 核心 |
|------|------|-----|-----------|------------|
| A | multigpt | mvp_24 | 80 / 90 | 拆解→串并联专家→汇总 |
| A | morngpt | mvp_28（26/27） | 80 / 90 | 17+ 垂直专家 + 流式多模态 |
| A | mornbusiness | mvp_19（相关 mvp_4） | 100 / 85 | 线索→获客→资本/博主对接 |
| B | mornclient | mvp_30/34（31/32） | 70 / 90 | 需求解析→构建矩阵→上架清单 |
| B | mornfront | mvp_22（相关 mvp_11） | 70 / 60 | UI 意图→组件编排→预览 ZIP |
| B | mornsite | mvp_8 | 60 / 90 | 站点图谱 / 扩展入口（可加 AI 找站） |
| C | morncontract | mvp_21 | 80 / 90 | 条款抽取→模板→签署·AI 免责 |
| C | morncoach | mvp_15 + mvp_25 | 80 / 90 | 定目标→题库→诊断→语音面试官 |
| C | mornspeaker | mvp_29 | 70 / 60（软著） | ASR→多目标语翻译→房间广播 |
| D | mornsocial | mvp_1 | 100 / 85 | 画像→匹配→破冰→缘分申诉 |
| D | mornxyz | mvp_2 | 70 / 90（软著） | 模糊意图→半径过滤→深链 |
| D | mornhome | mvp_13 | 70 / 20（软著） | 搜房/搜客 + 押金托管 |
| E | morntool | mvp_7 | 70 / 85 | 找工具→操控 App→批量 RPA |
| E | mornfake | mvp_5 + mvp_6 | 90 / 80 | 生成路由 + 多模态鉴伪 |
| E | mornchat | mvp_33 | 60 / 70 | 会话/频道 + 摘要/翻译/待办旁路 |

### 3.2 三款 TBD

| # | 产品 | 赛道延伸 | MVP | 下一刀 |
|---|------|----------|-----|--------|
| T1 | **mornpricer** | A/B 侧翼（价格智能） | mvp_3 → **mvp_35 PickGlobal** | 扩展免费层→B 端选品；利润确定性计算，Agent 只解释 |
| T2 | **mornjob** | C/D（招聘 Agent） | mvp_10（绑 mvp_15） | 先卖面试模拟/简历优化，再开 B 席位 |
| T3 | **mornstack** | B（全栈生成） | mvp_23 | 与 front/client 划界；安全默认 + 可上线门槛 |

---

## 4. MVP 1–35 全谱（GitHub `yuxuanzhouo3`）

| MVP | Repo 简述 | 落入产品线 |
|-----|-----------|------------|
| 1 | social match / PersonaLink | mornsocial |
| 2 | random interesting topic | mornxyz |
| 3 | compare product price | mornpricer |
| 4 | User Growth Analysis | mornbusiness（增长模块） |
| 5 | AI Generator 文图音视频 | mornfake（生成） |
| 6 | Deepfake Detector | mornfake（检测） |
| 7 | Auto Scripts / Multi Sender | morntool |
| 8 | 流行站点聚合 | mornsite |
| 9 | Data Security Tool | 安全工具线（与 1 曾互换叙事） |
| 10 | Job Platform ML Matching | mornjob |
| 11 | AI Auto Coder（Cursor-like） | mornfront 相关 |
| 12 | B2B Demand Matching | 获客/供需（business 侧翼） |
| 13 | Rent + Deposit AI lawyer | mornhome |
| 14 | Health Matching | 健康匹配（矩阵外储备） |
| 15 | AI Interviewer | morncoach / mornjob |
| 16 | Blog helper | 内容助手（储备） |
| 17 | Free APIs/VPN 聚合售卖 | 工具供给（储备） |
| 18 | StartUp Helper VC/市场匹配 | mornbusiness 侧翼 |
| 19 | Business Helper Tax/Hiring/Sell | **mornbusiness 主** |
| 20 | EQ Helper 人脉匹配 | 社交侧翼 |
| 21 | Document Signature | **morncontract** |
| 22 | Front End UI GPT | **mornfront** |
| 23 | FullStack GPT Builder | **mornstack** |
| 24 | Multi Prompt GPT | **multigpt** |
| 25 | AI Teacher | **morncoach** |
| 26–28 | mornGPT 服务组合→前端底座→全栈可下载 | **morngpt** |
| 29 | VoiceLink 语音翻译 | **mornspeaker** |
| 30 | mornClient App | mornclient |
| 31 | mornExtension | mornclient 承接 |
| 32 | mornMini 小程序 | mornclient 承接 |
| 33 | Enterprise / 多场景 Chat | **mornchat** |
| 34 | URL/一句话→App+扩展+小程序合一 | **mornclient 高阶** |
| 35 | PickGlobal 选品/比价/获客 | **mornpricer 演进** |

未并入「15+3」主叙事的侧翼 MVP：`9, 12, 14, 16, 17, 18, 20` —— 面试一笔带过即可。

---

## 5. MVP 1–35 结论：Agent CTO 三大件（Eval / Architecture / Harness）

> 15 款产品证明你在 **5 大热门 Agent 赛道**都打过 0→1；白板时任选 1 赛道 + 1 产品，用下面三件套推演 90 天。

### 5.1 Agent Eval（评测）+ 可观测性与单位经济学

- **目标函数三维门禁**：任务成功率 × 单位成本（token/工具调用）× 端到端时延；禁止只用「聊天轮次」当成功。
- **产品级评测集**：每赛道自带场景集（编排任务完成率、生成 App 可运行率、合同幻觉率、成聊率、提分/Offer、WER、鉴伪精确率/召回等）。
- **对抗与漂移**：mornfake 对抗样本压测；价源/题库/房源脏数据进版本化规则，禁止静默污染模型。
- **可证伪北极星**：CAC_qualified、7 日触达回复率、模考提分、托管开通率 —— 影子流量 / 小样本 A-B，拒绝体感。
- **工具链**：以 **Harbor**（`yuxuanzhouo3/harbor`）做 Agent/LM 评测与并行实验；复杂任务基准集发版前过门禁。
- **决策可观测**：每次 Agent 决策落库 —— prompt 版本、工具轨迹、token 成本、成功率；支持审计与复盘。
- **成本/延迟预算**：多专家路由下，小模型 / 大模型 / 规则引擎何时切换；缓存与批处理。
- **涨价追问标准答**：模型涨价 30% 时，用路由降级、语义缓存、配额 FEFO 与 SKU（席位/任务包/加油包）守住毛利，不先砍转化。

### 5.2 Architecture（架构）+ 全栈商用组件

- **Agent Runtime（公司级）**：任务图 DAG · 工具沙箱 · 记忆层（会话/租户/行业知识库）· human-in-the-loop。不是「页面里塞 Chat」。
- **Supervisor–Worker**：multigpt 代表形态 —— 拆解、串并联、失败重试、工具权限分级；硬限制防死循环烧钱。
- **平台层 vs 专家层**：路由/记忆/工具/计费/评测属平台；专家只交付 skill + 评测集（morngpt 17+ 可规模化）。
- **确定性内核 vs LLM**：金额/汇率/关税/押金状态机/合同 JSON Schema 由确定性引擎算；LLM 只解释、抽取、建议。
- **双栈抽象**：国内 CloudBase/微信 vs 国际 Supabase/Google OAuth —— 同一领域模型，避免业务逻辑分叉。
- **多端同一 Agent API**：权限与配额 FEFO 跨 Web / 小程序 / App / 桌面 / 扩展共用。
- **全栈商用验收**：登录 → 支付账本 → 核心功能 → AI Agent → Admin → Market → 客户端矩阵。
  - 支付：微信/支付宝 + Stripe/PayPal；分佣可对账、可开票、可风控。
  - Admin：租户隔离、审计、邀请码、举报 SLA、模型 A/B、滥用封禁。
  - 端优先级按场景切刀（比价 Chrome 先行；合同微信成交 + Web 编辑；效率工具桌面主战场）。

### 5.3 Harness（运行与实验挽具）

- **评测 Harness**：Harbor —— 任意 Agent + 数据集并行跑环境；支撑回归与 RL rollout。
- **执行 Harness**：工具白名单、动作可回放、一键撤销（morntool）；构建队列熔断（mornclient）。
- **发布 Harness**：prompt / 模型 / 模板三位一体版本号；灰度与策略开关在 Admin。
- **协作 Harness**：关键路径可人工审批与轨迹回放（销售演示 + 合规审计同一套日志）。

---

## 6. Agent CTO 面试作战卡（90 天白板模板）

先报赛道 A–E，再锁 1 个产品，按 Excel N 列三段答：

1. **内部 AI Native（约 30 天）** — Runtime + 评测门禁 + SOP→skill + 人工接管  
2. **PMF 验证（约 30–60 天）** — 可证伪指标 + 场景切刀 + 冷启动实验  
3. **产品商用（约 60–90 天）** — 登录支付账本 + Admin SLA + 多端 + 单位经济学  

**追问池（按赛道抽）**  
A：万级租户周报多智能体单位经济 · DeepSeek 冲击下专家工作流 ARPU · 模型涨价 30% 守毛利  
B：商店审核突变 · 生成代码「能跑不专业」退款 · 生成系统被打穿责任边界 · 新标签能力取消后的 Plan B  
C：合同欺诈证据链 · 只刷题不提分口碑 · 高峰推理排队降级  
D：匹配升点击却降成聊 · 无独家供给护城河 · 资金托管合规分期 · 海投泡沫抑制  
E：生成滥用熔断 · 系统收紧自动化权限备选 · 与飞书/Slack 差距下「不做什么」仍卖首批 50 家 · 源站反爬 48h 降级  

---

## 7. 简历段落（v2 · 可直接粘贴）

**MornScience｜创始人 / CEO / Agent CTO｜2023.08 – 至今**

- 从 0 到 1 创建公司并搭建技术团队；GitHub **mvp_1–mvp_35**（`yuxuanzhouo3`）+ `yzcmf/mornscience`；建立「需求挖掘 → Agent 产品化 → 多端上线 → 运营」闭环。
- 主导 **15 款 AI Agent 完成 0→1**，覆盖 **5 大热门赛道**：多智能体编排（multigpt/morngpt/mornbusiness）、Coding/端生成（mornclient/mornfront/mornsite）、垂直业务（morncontract/morncoach/mornspeaker）、匹配与双边市场（mornsocial/mornxyz/mornhome）、工具自动化与多模态（morntool/mornfake/mornchat）；另 **3 款 TBD**（mornpricer→PickGlobal、mornjob、mornstack）。
- 落地公司级 **Agent Runtime**（DAG / 工具沙箱 / 记忆 / HITL）与 **Harbor 评测 harness**（成功率×成本×时延门禁 + 决策轨迹落库）；统一登录、支付账本、Admin、多租户与 CN/Intl 双栈。
- 用可证伪北极星驱动 AI Native → PMF → 商用；具备模型涨价/路由降级、配额 FEFO、SKU 与毛利防守的实战话术与产品机制。

更完整简历正文见：`周宇轩简历(详细版_AgentCTO_v2).md` / `.docx`。

---

## 8. 图例

- ✓ 已实现或较完整 · △ 部分实现/明确规划 · ✗ 当前未见  
- 需求 / 商用分数来自 `市场需求挖掘1.xlsx`  
- 「15 + 3」与「5 大赛道」为简历与面试统一口径  

*整理：product.md 结论收敛为 Eval / Architecture / Harness 三大件；15 产品映射 5 大热门 Agent 面试赛道。*
