
def score(flop=1170, params=6.9):
    base_flop, base_params = 1170.0, 6.9
    return (flop / base_flop) + (params / base_params)


if __name__ == '__main__':
    candidates = [[1684, 15.4],
                  [588, 3.6],
                  [656, 3.4],
                  [656, 3.4],
                  [590, 4.5],
                  [634, 4.2],
                  [434, 5.4],
                  [1148, 4.7],
                  [1128, 5.3],
                  [1176, 5.1],
                  [1170, 6.9]
                  ]
    for candidate in candidates:
        print(score(candidate[0], candidate[1]))
