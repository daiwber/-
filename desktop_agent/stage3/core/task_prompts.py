task_prompts = {
    "用户登录": [
        "双击{app_name}应用图标打开应用",
        "等待{login_window}出现",
        "点击{account_login_tab}",
        "在{username_field}中输入用户名 '{username}'",
        "在{password_field}中输入密码 '{password}'",
        "点击{login_button}完成登录",
        "等待主界面加载完成"
    ],
    "网页搜索": [
        "双击{app_name}应用图标打开浏览器",
        "点击{address_bar}",
        "在{address_bar}中输入 'www.baidu.com'，并按回车",
        "等待页面加载完成",
        "在{search_box}中输入关键词 '{search_term}'，并按回车",
        "等待搜索结果加载"
    ]
}