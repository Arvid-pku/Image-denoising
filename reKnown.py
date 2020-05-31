def reKnow(orgp, denoised):
    [r, c, s] = orgp.shape
    for i in range(0, r):
        for j in range(0, c):
            if orgp[i][j][0] + orgp[i][j][1] + orgp[i][j][2] != 0:
                for k in range(0, s):
                    denoised[i][j][k]=orgp[i][j][k]
