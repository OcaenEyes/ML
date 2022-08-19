"""支持结果导出"""
import csv
import json


def exporter(directory, method, datasets):
    """支持结果导出"""
    if method.lower() == 'json':
        # 将json_dict转换为json样式的字符串
        json_string = json.dumps(datasets, indent=4)
        savefile = open('{}/exported.json'.format(directory), 'w+')
        savefile.write(json_string)
        savefile.close()

    if method.lower() == 'csv':
        with open('{}/exported.csv'.format(directory), 'w+') as csvfile:
            csv_writer = csv.writer(
                csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for key, values in datasets.items():
                if values is None:
                    csv_writer.writerow([key])
                else:
                    csv_writer.writerow([key] + values)
        csvfile.close()