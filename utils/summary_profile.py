def process_counts(total_params, total_mults, total_adds,
                   mul_bits, add_bits, param_bits=16., param_mb=False):
    # converting to Mbytes.
    if param_mb:
        total_params = int(total_params) / 8 / 1e6
    else:
        total_params = int(total_params) / param_bits / 1e6
    total_mults = total_mults * mul_bits / 16 / 1e6
    total_adds = total_adds * add_bits / 32 / 1e6
    return total_params, total_mults, total_adds


def _print_line(name, input_size, kernel_size, in_channels,
                out_channels, param_count, flop_mults, flop_adds, mul_bits,
                add_bits, base_str=None, quantizable=False):
    """Prints a single line of operation counts."""
    op_pc, op_mu, op_ad = process_counts(param_count, flop_mults,
                                              flop_adds, mul_bits, add_bits)
    # TODO: print quantizable
    output_string = base_str.format(
        name, str(quantizable), input_size, kernel_size, in_channels, out_channels, param_count,
        flop_mults, flop_adds, flop_mults + flop_adds)
    print(output_string)


def summary(file_name, output_file):
    _line_str = ('|{:40s}| {:15s}| {:10d}| {:13d}| {:13d}| {:13d}| {:15.3f}| {:10.3f}|'
                 ' {:10.3f}| {:10.3f}|')
    with open(file_name, 'r') as f:
        content = f.readlines()

    content = content[2:]
    last_block_name = 'data'
    input_size = 224
    total_params = 0
    total_mults = 0
    total_adds = 0
    cin = 3
    for i, line in enumerate(content):
        item_list = [item.strip() for item in line.split('|')][1:-1]
        block_name = item_list[0]
        if block_name[:5] != last_block_name:
            if i != 0:
                last_item_list = [item.strip() for item in content[i - 1].split('|')][1:-1]
                cout = int(last_item_list[5])
            else:
                cout = -1
            _print_line(last_block_name, input_size, -1, cin,
                        cout, total_params, total_mults, total_adds,
                        32, 16, base_str=_line_str, quantizable=False)
            last_block_name = block_name[:5]
            if item_list[2] == '':
                input_size = 1
            else:
                input_size = int(item_list[2])
            total_params = float(item_list[-4])
            total_mults = float(item_list[-3])
            total_adds = float(item_list[-2])
            if item_list[2] == '':
                cin = -1
            else:
                cin = int(item_list[4])
        else:
            total_params += float(item_list[-4])
            total_mults += float(item_list[-3])
            total_adds += float(item_list[-2])
        x = 2

    count = 1
    with open(output_file, 'w') as f:
        for line in content:
            if line[0] == '[':
                f.write(line)
                if count % 3 == 0:
                    f.write('-' * 40 + '\n')
                count += 1


if __name__ == '__main__':
    summary('./profile.txt', './summarized_profile.txt')
