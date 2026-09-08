## Purpose

为对话生成与文本嵌入提供统一的、可替换的模型接入契约，使使用者能按预算、可用性与数据合规要求自行选择模型供应方，而不影响系统其余部分的行为。

## ADDED Requirements

### Requirement: Provider 契约

系统 SHALL 定义 Chat 与 Embedding 两类 provider 契约。任何符合契约的实现 MUST 可被系统使用而无需修改调用方代码。

#### Scenario: 更换 provider 不改调用方
- **WHEN** 将配置中的 chat provider 从一家切换为另一家
- **THEN** 问答链路无需任何代码改动即可继续工作

#### Scenario: 未知 provider 名称
- **WHEN** 配置中指定了一个未注册的 provider 名称
- **THEN** 系统在启动时报错并列出所有可用的 provider 名称，而非在首次调用时才失败

### Requirement: 内置实现

系统 SHALL 内置 OpenAI、DeepSeek 与本地嵌入模型三种实现。Chat 与 Embedding 的 provider MUST 可被独立配置为不同供应方。

#### Scenario: 混合搭配
- **WHEN** 配置 chat 使用 DeepSeek、embedding 使用本地模型
- **THEN** 系统按该组合工作，生成走 DeepSeek，嵌入在本地完成且不产生外部网络调用

#### Scenario: 离线可用的嵌入
- **WHEN** 配置 embedding 为本地模型且无外网连接
- **THEN** 索引构建与检索仍可正常完成

### Requirement: 凭据校验与保护

系统 SHALL 在启动时校验所选 provider 所需凭据是否齐备。凭据 MUST NOT 出现在任何日志、错误信息或 API 响应中。

#### Scenario: 缺少必需凭据
- **WHEN** 配置的 provider 需要 API key 而环境中未提供
- **THEN** 系统在启动时以明确信息中止，指出缺少哪个环境变量名

#### Scenario: 凭据不出现在错误信息中
- **WHEN** provider 因凭据无效返回认证错误
- **THEN** 系统向上报告认证失败，且信息中不包含该凭据的任何字符

### Requirement: 失败处理

对可重试的错误（限流、超时、5xx），provider 调用 SHALL 按配置退避重试；重试耗尽后 MUST 向调用方返回可区分的错误类型，而非静默返回空结果。

#### Scenario: 限流后重试成功
- **WHEN** provider 返回限流错误，重试后成功
- **THEN** 调用方获得正常结果

#### Scenario: 重试耗尽
- **WHEN** 重试次数达到配置上限仍失败
- **THEN** 调用方收到明确标识为 provider 失败的错误
- **AND** 系统不返回空字符串或空向量冒充成功结果
