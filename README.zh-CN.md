# HierTune: Two-Level Search-Space Revision for LLM-Assisted Database Tuning

[中文](README.zh-CN.md) · [English](README.md)

## 1. HierTune_v1：论文方法原文

代码目录：`hiertune_v1/src/cognitune/`。V2的代码与运行入口独立保存在 `hiertune_v2/`。

### Overview

As illustrated in the framework overview, HierTune adaptively revises knowledge-guided search-space priors through coordinated local and global updates. 
Multiple domain experts organize active knobs into sparse, each maintaining an independent recommended region.
At the micro level, HierTune uses execution feedback to revise local regions and knob combinations without directly changing the global active set.
At the macro level, it periodically reassesses the active  knob and overall reference ranges using estimated removal losses and updated expert recommendations. 
To explore the evolving subspaces, HierTune progressively allocates candidates by combining these estimates with region scale and current coverage, adapting search effort to spatial revision and exploration progress. 
Candidates from all spaces enter a unified pool, where qLogNEI selects one configuration for real execution. The resulting feedback updates the shared Gaussian process (GP) and informs subsequent revisions, forming a closed tuning loop.

### Multi-Expert Knob Initialization

Database knobs affect different performance mechanisms. We therefore use four domain experts covering memory and cache, concurrency and execution, storage and write-ahead logging (WAL), and query optimization and workload analysis. Following MCTuner, a manager agent identifies the relevant experts for each knob and assigns weights based on the workload, hardware environment, and tuning objective. These nonnegative weights sum to one for each knob. Experts use parameter documentation and initial observations, together with the hardware and workload information, to score knob importance on a 1--100 scale and recommend value ranges. The system aggregates their scores through a weighted average to rank the knobs.

The system selects the top 20 knobs to form the initial active set $A_0$, subject to parameter dependencies, resource constraints, and domain coverage. Expert recommendations also define the overall reference ranges $\mathcal R_0^{\mathrm{ref}}$. Unselected knobs remain in $K$ for reconsideration during later macro-level revisions.

### Sub-Tuning Spaces

The effectiveness of a knob depends on its interaction with other knobs, making a single high-dimensional search space difficult to revise using limited execution feedback. To localize exploration and subsequent revisions, HierTune organizes active knobs into sparse groups that can be adjusted jointly, forming sub-tuning spaces

$$
    h=(C_h,R_{h,t}),
    \qquad
    \varnothing\neq C_h\subseteq A_t,
    
$$

where $t$ is the current tuning iteration, $A_t$ is the active knob set, $C_h$ is the group of jointly adjusted knobs, and $R_{h,t}$ is the recommended region maintained independently for this combination. By default, each expert maintains at most two regular sub-tuning spaces, each containing at most four knobs. Experts propose combinations and regions based on the overall reference ranges, current configuration, and execution history.

Independent regions are central to two-level revision. The same knob may have different effective values in different combinations. Subspaces that share a knob therefore maintain separate local ranges for it. Micro-level revision changes only the corresponding subspace without overwriting other combinations or the overall reference ranges. This allows local priors to evolve independently while retaining shared execution feedback for performance modeling.

Candidates are built upon the current best configuration $a_t$. Only knobs within the group are modified, and necessary activation dependencies are explicitly included in the group. Numerical knobs designated for logarithmic scaling are log-normalized; other numerical knobs are linearly normalized. Candidates are sampled within the recommended range using a truncated normal distribution centered at the recommended value, with a $10\%$ probability of global exploration. Integer values are rounded to their legal granularity. Enumeration and Boolean parameters are mainly sampled from recommended options, while retaining a chance to explore other legal options. Sampled configurations and explicit joint patches from experts form the candidates, which undergo validity checks, deduplication, and filtering of previously evaluated configurations.
These independently maintained subspaces provide the units for the local revisions described next.

### Micro-Level Spatial Revision

Execution feedback from a specific knob combination does not directly represent the improvement potential of these knobs in other combinations. HierTune therefore revises recommended regions and knob combinations within individual sub-tuning spaces while keeping the global active set and reference ranges unchanged. A shared Gaussian Process (GP) models the relationship between configurations and performance utility using the execution history $\mathcal D_t=\{(x_s,y_s)\}_{s=1}^{t}$, where $y_s$ is the observed utility of configuration $x_s$.

The GP provides a joint posterior distribution for candidate configurations and the reference configuration. Each draw produces a set of possible utility values while preserving correlations between configurations. Let $F$ denote one draw, with $F(x)$ representing the utility of configuration $x$ in that draw. For a candidate set $B$ held fixed during comparison, we define

$$
    Z_t(B;F)=
    \left[\max_{x\in B}F(x)-F(a_t)-\delta\right]_+,
    \qquad
    V_t(B)=\mathbb E[Z_t(B;F)\mid\mathcal D_t].
    
$$

Here, $a_t$ is the current best configuration, $[v]_+=\max(v,0)$, and $\delta$ is the minimum improvement threshold. This estimate measures whether the candidate set offers improvement over the reference configuration and the expected magnitude of that improvement. We use $\delta=0.01$ and 256 joint posterior draws by default. These draws involve only model computation and require no additional database executions. Different proposals use the same reference configuration and candidate count.

To locate promising values, the system generates valid probe configurations inside and outside the recommended region, forming $\mathcal P_h$. Internal probes cover the current region, while external probes combine boundary expansion with sampling over the wider legal domain. Knobs outside the group remain fixed at $a_t$. Each posterior draw identifies the best probe configuration and adds its improvement beyond the threshold to the corresponding location. The accumulated values are normalized into weights that reflect both winning frequency and improvement magnitude. If the total improvement is zero, the original region is retained.

The system selects the valid probe configuration with the smallest total weighted distance to the improvement locations as the new center, preserving its joint knob values. For numerical knobs, distances from improvement locations to the new center are normalized by the original half-widths, taking the maximum across dimensions. Their weighted 80th percentile determines the half-width scaling factor, restricted to $[0.5,2]$. Concentrated improvement locations refine the region, dispersed locations expand coverage, and center changes shift the search location. Mixed-type distances account for numerical differences and categorical mismatches. For categorical knobs, experts propose new legal preference sets based on the improvement locations.

To determine whether the proposed region should replace the original one, the system generates equal numbers of candidates, forming $B_h'$ and $B_h$, and computes

$$
    Q_t(B_h',B_h)=
    \Pr\!\left[
    \max_{x\in B_h'}F(x)-\max_{x\in B_h}F(x)>\delta
    \;\middle|\;\mathcal D_t
    \right].
    
$$

Whereas the opportunity estimate measures improvement over the reference configuration, this comparison requires posterior support for an advantage greater than $\delta$ over the original region. The revision is accepted when $Q_t(B_h',B_h)>\rho$, with $\rho=0.7$ by default. The accepted region applies only to the current knob combination and does not change the ranges of other subspaces.

When a sub-tuning space produces no improvement over three consecutive actual evaluations, the system requests a review from the relevant expert. The expert receives the current combination, recommended region, full execution history, and information about improvement opportunities and candidate coverage inside and outside the region. It then considers whether to continue exploration, propose a range revision, or suggest alternative combinations. If a combination update is needed, the expert proposes several new knob groups from the active set. The system generates equal numbers of candidates for the proposed and existing groups maintained by that expert and evaluates their improvement opportunities using $V_t$. The highest-scoring groups, up to two, and their recommended regions become the expert's maintained sub-tuning spaces. Thus, each expert maintains at most two knob groups, and new proposals compete with existing groups rather than replacing them directly.

### Macro-Level Revision

Local improvements reflect performance under specific combinations and cannot directly justify changes to the overall prior. HierTune therefore jointly reviews the active set and overall reference ranges every $H=12$ adaptive configuration evaluations. The review includes only currently active knobs whose values have been changed in at least three actual configuration evaluations after the first ten initialization evaluations. This avoids removing knobs solely because they lack observed contributions or sufficient exploration.

Let $S=(A,\mathcal R^{\mathrm{ref}},\mathcal H)$ denote the search state, consisting of the active knob set, overall reference ranges, and sub-tuning space set. Following fixed rules, the system generates equal numbers of candidates for each expert to form a state comparison set $B(S)$. It reuses the micro-level opportunity estimate and defines $\Phi_t(S)=V_t(B(S))$.

After removing a knob or a necessary dependency group $U$, its values are fixed at those in $a_t$, and the remaining state $S^{-U}$ is rebuilt using the same GP. The removal loss is $\mathcal L_t(U)=\Phi_t(S)-\Phi_t(S^{-U})$. A smaller loss indicates that the current search depends less on this object and that its estimated marginal value is lower under the current active set and reference configuration. Removal must satisfy both conditions:

$$
    \mathcal L_t(U)\leq\gamma_{\mathrm{rm}},
    \qquad
    \Pr\!\left[
    Z_t(B(S^{-U});F)
    \geq Z_t(B(S);F)-\gamma_{\mathrm{rm}}
    \;\middle|\;\mathcal D_t
    \right]>\rho.
    
$$

Here, $\gamma_{\mathrm{rm}}=\delta$ is the acceptable opportunity loss, and $\rho=0.7$. The two conditions constrain the expected removal loss and the posterior probability that the loss remains within tolerance. The original and removal states use equal numbers of candidates. Common random numbers are used for shared coordinates where possible to reduce sampling differences.

The system selects removal objects sequentially by removal loss and re-evaluates the remaining choices after each removal. At most five knobs are replaced per period. Each expert then re-ranks inactive knobs using the latest execution feedback and recommends at most six candidate knobs. The initialization weighting procedure determines their entry priority. Experts also update the overall reference ranges and new or affected subspaces, while other combinations retain their local regions. Changes to the active set do not alter the GP's full configuration encoding, so historical values of removed knobs remain available for performance modeling.

### Candidate Allocation and Closed-Loop Execution

As micro- and macro-level revisions change knob combinations and recommended regions, the improvement opportunities and sampling needs of individual sub-tuning spaces also evolve. HierTune therefore progressively allocates candidates according to improvement opportunities and coverage requirements.

Each sub-tuning space first generates four initial candidates. To account for database execution noise, the shared GP computes noisy expected improvement (NEI). In each joint posterior draw, the candidate's utility is compared with the highest utility among previously evaluated configurations. Positive differences are retained and then averaged across draws. Unlike ordinary expected improvement with a fixed best-value baseline, this calculation also accounts for uncertainty in the performance of previously evaluated configurations.

Following the regional averaging idea of REI, the system uses the candidates' mean NEI, $P_{h,t}$, to estimate the improvement opportunity of the current candidate distribution. Unlike $V_t(B)$, which measures the best improvement of a candidate set for structural comparison, $P_{h,t}$ summarizes average pointwise improvement. The system uses the relative score $\widetilde P_{h,t}=P_{h,t}/\max_{g\in\mathcal H_t}P_{g,t}$, setting it to zero when all scores are zero. This normalization preserves relative differences between spaces and prevents changes in the absolute NEI scale from altering the balance between improvement and coverage.

A finite candidate set may still miss promising locations. The system therefore also considers region scale, dimensionality, and candidate count. Let $n_{h,t}\geq4$ be the current candidate count, $d_h=|C_h|$ the knob count, and $L_{h,t}$ the maximum distance between legal configurations in the recommended region. This distance is measured in GP encoding coordinates scaled by their length scales. Numerical and ordered categorical values use normalized coordinates, while Boolean and unordered categorical values use one-hot coordinates. Knobs outside the group remain fixed at $a_t$. The next candidate is assigned to

$$
    \kappa_{h,t}(n)=L_{h,t}n^{-1/d_h},
    \qquad
    h^*\in\arg\max_{h\in\mathcal H_t}
    (\lambda+\widetilde P_{h,t})
    \left[
    \kappa_{h,t}(n_{h,t})-\kappa_{h,t}(n_{h,t}+1)
    \right].
    
$$

Here, $\mathcal H_t$ is the current set of sub-tuning spaces. We use $\lambda=0.2$ by default to retain a base coverage weight when predicted improvement is zero. The quantity $\kappa_{h,t}(n)$ is a geometric approximation based on region scale, knob count, and candidate count, rather than an actual coverage error. The rule prioritizes spaces with higher relative improvement opportunities where an additional candidate can still substantially reduce the remaining coverage scale. Scores, normalized values, and candidate counts are updated after each generation step. With fixed scores and regions, the priority of further investment decreases as more candidates are generated.

After candidate generation, all valid, unique, and previously unevaluated configurations enter a unified pool, from which qLogNEI selects one configuration for execution. The observation updates the shared GP, current best configuration, and subspace exploration state, providing feedback for subsequent local and global revisions. The revised subspaces then guide candidate generation in the next round. This mechanism combines relative improvement opportunities with coverage requirements and allows candidates from multiple subspaces to compete for execution.

### 超参数

| Parameter | Default |
|---|---|
| Domain experts / initial active knobs | 4 / 20 |
| Subspaces per expert / knobs per subspace | At most 2 / at most 4 |
| Global sampling probability | 0.1 |
| Minimum improvement $\delta$ / posterior draws | 0.01 / 256 |
| Width quantile / scale factor bounds | 0.8 / [0.5, 2] |
| Acceptance probability $\rho$ | 0.7 |
| Local stagnation review | 3 consecutive actual local evaluations |
| Macro review interval | 12 adaptive evaluations |
| Macro eligibility / replacement cap | 3 actual changes / at most 5 knobs |
| Inactive nominations per expert | At most 6 |
| Acceptable removal loss $\gamma_{\mathrm{rm}}$ | $\delta=0.01$ |
| Initial candidates per subspace / allocation weight $\lambda$ | 4 / 0.2 |

## 2. HierTune_v2

V2的方案是大小轮均由大模型依据执行证据决策：每12个自适应轮次开展宏观审查，决定活动旋钮的保留与替换；每轮专家结合共享GP证据、完整执行历史、当前配置、负载和可用机器状态，决定旋钮组合及推荐范围。范围由专家提出，程序仅作合法边界截断与约束检查。候选仍由共享代理模型选择一个进行执行。

## 3. 复现环境与V2启动

需要 Linux、Python 3.10–3.12（推荐3.11）、PostgreSQL 14.9、JDK 23 和 BenchBase。Python依赖由 `pyproject.toml` 声明。数据库安装路径为 `/opt/cognitune-experiment/postgresql-14.9`，数据目录为 `/var/lib/cognitune-experiment/data`，端口为55432；需预先配置具有基准数据库管理权限的 `benchadmin` 用户。

BenchBase源码置于 `.runtime/benchbase`，使用提交 `33c00473807ebd49304d114a6d769d2d2b2bbb34`，先构建PostgreSQL发行包并解压至 `target/benchbase-postgres`。运行入口会构建测量窗口统计补丁。YCSB每次评测使用16终端、15秒预热、60秒测量；先实测三次默认配置，再重新测量保留的十个共同初始化配置。初始化为历史IID设计，不能称为LHS。

请在本机 `.runtime/cognitune.env` 设置 `DEEPSEEK_API_KEY`、`PGPASSWORD` 等凭据，禁止提交此文件。模型请求名为 `deepseek-flash`，可用模型以服务端为准。真实实验使用数据库互斥锁，不要同时启动其他负载。

以下仅给V2的现有每轮专家入口；该历史入口逐个请求专家，尚未实现上文所述完整大小轮LLM修订协议与并行调用，不能作为完整V2实现的验证结果。

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python -u hiertune_v2/run_ycsb.py --budget 50 --output artifacts/hiertune_v2_ycsb50
```

发布数据仅包括指定的四张PNG、行为分析对应的YCSB轨迹及 `gp20_small_candidate_grid_v1`。行为图的展示调整记录保留在YCSB目录的 `figure_display_edits.json`，不应将展示配额当作未经修改的执行日志。
