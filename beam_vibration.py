import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

def main():
    # パラメータ設定
    L = 1               # ビームの長さ
    N = 100             # 空間分割数
    x = np.linspace(0, L, N)  # 空間座標
    dx = x[1] - x[0]          # 空間ステップサイズ

    # 質量値をユーザーから入力
    while True:
        try:
            m_values_input = input("質量値をカンマ区切りで入力してください（例: 1,10,100）: ")
            m_values = [float(m.strip()) for m in m_values_input.split(",")]
            if m_values:
                break
            else:
                print("少なくとも一つの質量値を入力してください。")
        except ValueError:
            print("無効な入力です。数値をカンマ区切りで入力してください。")

    # 点質量の位置をユーザーから入力
    while True:
        try:
            load_position_input = input(f"点質量の位置を入力してください（0から{N}の整数）: ")
            load_position = int(load_position_input)
            if 0 <= load_position <= N:
                break
            else:
                print(f"0から{N}の整数を入力してください。")
        except ValueError:
            print("無効な入力です。整数を入力してください。")

    def delta_func(N, position):
        # デルタ関数を定義（点質量の位置で1/dx、それ以外は0）
        delta = np.zeros(N)
        delta[position] = 1 / dx
        return delta

    # 点質量のデルタ関数
    delta = delta_func(N, load_position)

    # 4次導関数行列の構築（有限差分法）
    coeff = 1 / dx**4
    diagonals = [
        np.full(N, 6) * coeff,       # Main diagonal
        np.full(N-1, -4) * coeff,    # First off-diagonals
        np.full(N-2, 1) * coeff      # Second off-diagonals
    ]
    D4 = diags(diagonals, [0, 1, 2], shape=(N, N)).toarray()
    D4 += D4.T  # Make the matrix symmetric

    # 境界条件の適用（固定端）
    D4[0, :] = D4[-1, :] = 0
    D4[:, 0] = D4[:, -1] = 0

    # 初期条件の定義：三角形状の変位分布
    def initial_conditions():
        w0 = np.zeros(N)
        # 左側の線形領域
        w0[:load_position+1] = np.linspace(0, 1, load_position+1)
        # 右側の線形領域
        w0[load_position:] = np.linspace(1, 0, N - load_position)
        w1 = np.zeros(N)  # 初期速度（静止）
        return np.concatenate([w0, w1])

    # ビームの運動方程式の定義（修正済み）
    def beam_vibration(t, y, m):
        w = y[:N]      # 変位
        w_t = y[N:]    # 速度
        dw_dt = w_t    # 変位の時間微分は速度
        dw_tt = -D4 @ w / (1 + m * delta)   # 運動方程式
        dw_tt[0] = dw_tt[-1] = 0  # 境界条件の適用
        return np.concatenate([dw_dt, dw_tt])

    # 時間設定
    t_span = (0, 0.005)               # 時間範囲を長くする（例: 0.005秒まで）
    t_eval = np.linspace(*t_span, 500)  # 評価時刻の配列を増やす

    # プロットとアニメーションのセットアップ
    fig, ax = plt.subplots(figsize=(10, 6))

    # 初期条件ラインのプロット（初期時刻の変位分布）
    y0 = initial_conditions()
    w0_initial = y0[:N]
    initial_line, = ax.plot(x, w0_initial, 'k--', lw=1, label='Initial shape')

    line, = ax.plot(x, np.zeros(N), 'b-', lw=2, label='Current displacement')
    ax.set_xlim(0, L)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel('x')
    ax.set_ylabel('Displacement w(x, t)')
    ax.set_title('Beam Displacement Animation')
    ax.legend()

    time_template = 'Time = {:.2f} s, Mass = {}'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    def animate(i, sol, m):
        line.set_ydata(sol.y[:N, i])
        time_text.set_text(time_template.format(sol.t[i], m))
        return line, time_text

    # アニメーションを保存するかどうかユーザーに確認
    save_option = input("アニメーションを保存しますか？ (y/n): ").lower()

    import os  # ファイル存在確認のためにインポート

    for m in m_values:
        print(f"m={m}の計算をしています！")
        y0 = initial_conditions()
        # 微分方程式の数値解法
        sol = solve_ivp(beam_vibration, t_span, y0, t_eval=t_eval, args=(m,), method='RK45')

        ani = FuncAnimation(fig, animate, frames=len(t_eval), fargs=(sol, m), interval=20, blit=True)

        if save_option == 'y':
            filename = f'beam_animation_mass_{m}.mp4'
            # ファイルの存在確認と上書き確認
            if os.path.exists(filename):
                overwrite = input(f"{filename} は既に存在します。上書きしますか？ (y/n): ").lower()
                if overwrite != 'y':
                    print(f"{filename} の保存をスキップしました。")
                    continue
            # アニメーションの保存
            ani.save(filename, fps=50, dpi=300, extra_args=['-vcodec', 'libx264'])
            print(f"アニメーションを {filename} として保存しました。")

    print("すべての計算が終了しました！")

if __name__ == "__main__":
    main()