import { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend,
  ScatterChart, Scatter, Cell,
} from 'recharts'
import { AgentHistoryPoint } from '../store/simStore'

interface Props {
  history: AgentHistoryPoint[]
}

const EMOTIONS = ['悲伤', '愤怒', '恐惧', '平静', '满足', '快乐', '兴奋']

function emotionIndex(e: string): number {
  const i = EMOTIONS.indexOf(e)
  return i >= 0 ? i : EMOTIONS.length
}

const CHART_MARGIN = { top: 4, right: 8, left: -20, bottom: 0 }

const tooltipStyle = {
  backgroundColor: '#1f2937',
  border: '1px solid #374151',
  borderRadius: 4,
  fontSize: 11,
  color: '#d1d5db',
}

export function AgentCharts({ history }: Props) {
  const emotionData = useMemo(
    () => history.map((p) => ({ tick: p.tick, val: emotionIndex(p.emotion), emotion: p.emotion })),
    [history],
  )
  const votingData = useMemo(
    () => history.filter((point) => point.opinion_voting_choice_counts !== null),
    [history],
  )
  const votingOptions = useMemo(
    () => Array.from(new Set(votingData.flatMap((point) => Object.keys(point.opinion_voting_choice_counts ?? {})))),
    [votingData],
  )
  const votingColors = ['#22c55e', '#ef4444', '#eab308', '#06b6d4', '#f97316', '#ec4899']

  if (history.length === 0) {
    return <div className="text-xs text-gray-500 text-center py-4">暂无历史数据</div>
  }

  return (
    <div className="space-y-3">
      {/* Satisfaction */}
      <div>
        <div className="text-xs text-gray-400 mb-1">满足度 (Satisfaction)</div>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={history} margin={CHART_MARGIN}>
            <XAxis dataKey="tick" tick={false} />
            <YAxis domain={[0, 100]} tick={{ fontSize: 9, fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v: number, name: string) => [v.toFixed(1), name]}
              labelFormatter={(l) => `t=${l}`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Line type="monotone" dataKey="satiety" name="饱食" stroke="#60a5fa" dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="relax" name="放松" stroke="#34d399" dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="money" name="金钱" stroke="#fbbf24" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Demand */}
      <div>
        <div className="text-xs text-gray-400 mb-1">急迫度 (Urgency)</div>
        <ResponsiveContainer width="100%" height={90}>
          <LineChart data={history} margin={CHART_MARGIN}>
            <XAxis dataKey="tick" tick={false} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 9, fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v: number, name: string) => [v.toFixed(3), name]}
              labelFormatter={(l) => `t=${l}`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Line type="monotone" dataKey="satiety_urgency" name="饱食急迫" stroke="#93c5fd" strokeDasharray="4 2" dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="relax_urgency" name="放松急迫" stroke="#6ee7b7" strokeDasharray="4 2" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 当前智能体的连续观念值由列表选择结果决定。 */}
      <div>
        <div className="text-xs text-gray-400 mb-1">观念变化 (Opinion)</div>
        <ResponsiveContainer width="100%" height={110}>
          <LineChart data={history} margin={CHART_MARGIN}>
            <XAxis dataKey="tick" tick={false} />
            <YAxis domain={[-1, 1]} ticks={[-1, 0, 1]} tick={{ fontSize: 9, fill: '#9ca3af' }} />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(v: number) => [v.toFixed(3), 'opinion']}
              labelFormatter={(l) => `t=${l}`}
            />
            <Line type="monotone" dataKey="opinion" name="观念" stroke="#a78bfa" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Opinion voting */}
      <div>
        <div className="text-xs text-gray-400 mb-1">观念投票 (LLM Voting)</div>
        {votingData.length === 0 ? (
          <div className="h-[90px] flex items-center justify-center text-xs text-gray-500">暂无投票数据</div>
        ) : (
          <ResponsiveContainer width="100%" height={110}>
            <LineChart data={votingData} margin={CHART_MARGIN}>
              <XAxis dataKey="tick" tick={false} />
              <YAxis domain={[0, 'auto']} allowDecimals={false} tick={{ fontSize: 9, fill: '#9ca3af' }} />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v: number, name: string) => [v, name]}
                labelFormatter={(l) => `t=${l}`}
              />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              {votingOptions.map((option, index) => (
                <Line
                  key={option}
                  type="monotone"
                  dataKey={(point: AgentHistoryPoint) => point.opinion_voting_choice_counts?.[option] ?? 0}
                  name={option}
                  stroke={votingColors[index % votingColors.length]}
                  dot
                  strokeWidth={1.5}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Emotion */}
      <div>
        <div className="text-xs text-gray-400 mb-1">情绪 (Emotion)</div>
        <ResponsiveContainer width="100%" height={80}>
          <ScatterChart margin={CHART_MARGIN}>
            <XAxis dataKey="tick" type="number" tick={false} name="tick" />
            <YAxis
              dataKey="val"
              type="number"
              domain={[-0.5, EMOTIONS.length + 0.5]}
              tickFormatter={(v: number) => EMOTIONS[v] ?? ''}
              tick={{ fontSize: 9, fill: '#9ca3af' }}
              width={36}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(_v: number, _name: string, props: { payload?: { emotion?: string } }) =>
                [props.payload?.emotion ?? '', '情绪']
              }
              labelFormatter={(l) => `t=${l}`}
            />
            <Scatter data={emotionData} name="情绪">
              {emotionData.map((entry, i) => (
                <Cell key={i} fill="#f472b6" />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
