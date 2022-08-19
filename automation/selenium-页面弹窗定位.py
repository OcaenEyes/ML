from selenium import webdriver

driver = webdriver.Firefox()
alert = driver.switch_to.alert

# 查看alert中的文字
print(alert.text)

# 点击确定
alert.accept()

# 点击取消（如果有）
alert.dismiss()
