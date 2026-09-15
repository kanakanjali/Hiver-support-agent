"""
Add reference replies to the golden set for ROUGE-L scoring.
Reference replies are based on how AppleSupport actually responds on Twitter:
concise, empathetic, and action-oriented.
"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Load existing golden set
with open(config.GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# Reference replies for a representative subset (at least 50 examples)
# These are modelled after real AppleSupport Twitter reply patterns.
REFERENCE_REPLIES = {
    "gs_001": "We're sorry to hear about this! For a cracked screen, we'd recommend visiting an Apple Store or an Apple Authorized Service Provider for an assessment. You can set up an appointment here: https://getsupport.apple.com ^KA",
    "gs_002": "That sounds frustrating. A loose charging port may need a repair. Please set up a service appointment at https://getsupport.apple.com so a technician can take a look. ^KA",
    "gs_003": "We understand the concern. Some warmth during use is normal, but let's check a few things. DM us your iPhone model and iOS version, and we'll help troubleshoot. ^KA",
    "gs_004": "We'd like to help with that! Try checking Settings > Sounds & Haptics and make sure nothing is blocking the speaker. If the issue persists, DM us and we'll look into next steps. ^KA",
    "gs_005": "We're sorry about that. You can use AssistiveTouch as a temporary workaround: Settings > Accessibility > Touch > AssistiveTouch. For a repair, visit https://getsupport.apple.com. ^KA",
    "gs_009": "That does sound unusual. Try restarting your iPhone first: hold Side + Volume Down, slide to power off, then turn back on. If it continues, DM us your model and we'll dig deeper. ^KA",
    "gs_010": "Let's help with that! Try these steps: 1) Clean the charging contacts 2) Try a different cable 3) Reset your AirPods by holding the setup button for 15 seconds. DM us if it persists. ^KA",
    "gs_015": "Fan noise can sometimes be caused by background processes. Open Activity Monitor to check if anything is using high CPU. Also make sure your Mac has room to ventilate. DM us if it continues! ^KA",
    "gs_019": "A lint buildup can definitely cause charging issues! Try gently cleaning the port with a wooden or plastic toothpick (avoid metal). Be careful not to damage the contacts. Let us know how it goes! ^KA",
    "gs_021": "We understand how frustrating that can be after an update. Try a force restart: quickly press Volume Up, then Volume Down, then hold the Side button until you see the Apple logo. DM us if it continues. ^KA",
    "gs_022": "Let's try getting your Mac into Recovery Mode. Restart it and immediately hold Command+R until you see the Apple logo. From there you can try Disk Utility or reinstall macOS. DM us for step-by-step help. ^KA",
    "gs_023": "Sorry about the crashes! Try deleting and reinstalling FaceTime, then restart your device. Also make sure you're on the latest iOS version. If it still crashes, DM us your details. ^KA",
    "gs_025": "We've heard about this and we're sorry it caused trouble! Make sure your Bedtime alarm is set up correctly in the Clock app (not just the Health app). Check Settings > Sounds & Haptics > Ringer volume too. ^KA",
    "gs_030": "Sorry about the pairing trouble! On your iPhone, go to Settings > Bluetooth, tap the (i) next to your Watch, and select 'Forget This Device.' Then re-pair from the Watch app. DM us if you need help. ^KA",
    "gs_049": "We understand the frustration. Try force-restarting: quickly press Volume Up, Volume Down, then hold Side button until Apple logo appears. If the boot loop continues, you may need DFU mode — DM us and we'll walk you through it. ^KA",
    "gs_050": "We're so sorry to hear about your photos. They may still be recoverable from iCloud. Check Settings > [Your Name] > iCloud > Photos to see if iCloud Photos is on. Also check Recently Deleted. DM us for more help. ^KA",
    "gs_055": "Locked accounts can be resolved! Visit iforgot.apple.com to start the recovery process. If you don't have access to your trusted devices or phone number, DM us and we'll help with next steps. ^KA",
    "gs_056": "We understand how stressful that is. Visit iforgot.apple.com and choose 'Forgot Apple ID or password.' If your recovery email is no longer accessible, you can start an account recovery. DM us for guidance. ^KA",
    "gs_060": "We can help you find it! Go to iforgot.apple.com and click 'look it up.' You can also check Settings on any device where you're signed in. DM us if you need more help. ^KA",
    "gs_073": "We'd like to help with that charge. You can request a refund at reportaproblem.apple.com. Sign in, find the purchase, and select 'Report a Problem.' DM us if you need further assistance. ^KA",
    "gs_074": "Sure thing! Go to Settings > [Your Name] > Subscriptions, tap Apple Music, then tap Cancel. Your subscription will remain active until the end of the billing period. ^KA",
    "gs_078": "The 200GB iCloud+ plan is $2.99/month. You can upgrade in Settings > [Your Name] > iCloud > Manage Account Storage > Change Storage Plan. ^KA",
    "gs_079": "Of course! Go to Settings > [Your Name] > iCloud > Manage Account Storage > Change Storage Plan and select 200GB. The change takes effect immediately. ^KA",
    "gs_083": "You can check your subscriptions at Settings > [Your Name] > Subscriptions. This shows all active and expired subscriptions so you can cancel any you don't need. ^KA",
    "gs_091": "Great question! You can use Quick Start to transfer everything wirelessly. Place both iPhones near each other, follow the on-screen prompts, and it'll transfer your data, settings, and apps. ^KA",
    "gs_092": "Go to Settings > General > About — you'll see the serial number listed there. You can also find it on the original box or in your Apple ID account page. ^KA",
    "gs_094": "Command + Shift + 3 captures the full screen. Command + Shift + 4 lets you select an area. Command + Shift + 5 gives you more options including screen recording! ^KA",
    "gs_095": "We recommend using iCloud Backup! Go to Settings > [Your Name] > iCloud > iCloud Backup > Back Up Now. Make sure you're on Wi-Fi and plugged in. ^KA",
    "gs_097": "Yes! It's called Sidecar. Make sure both devices are signed into the same Apple ID, then on your Mac go to System Settings > Displays > Add Display and select your iPad. ^KA",
    "gs_107": "You can check your battery health at Settings > Battery > Battery Health & Charging. It shows Maximum Capacity and Peak Performance Capability. ^KA",
    "gs_113": "Sure! Open the Notes app, tap the camera icon, and select 'Scan Documents.' Position the document in view and your iPhone will auto-capture it. ^KA",
    "gs_115": "Let's try some steps! Sign out of the App Store (Settings > [Your Name] > Media & Purchases > Sign Out), restart your device, then sign back in. Also check your internet connection. ^KA",
    "gs_119": "Nope! Apps you've purchased are always linked to your Apple ID. Go to the App Store > your profile > Purchased, and you can redownload it for free. ^KA",
    "gs_129": "We'd like to help! Try resetting your network settings: Settings > General > Transfer or Reset > Reset > Reset Network Settings. You'll need to re-enter Wi-Fi passwords. DM us if it persists. ^KA",
    "gs_130": "Let's try re-pairing. Go to Settings > Bluetooth, tap 'Forget This Device' for your car, then restart both your iPhone and car system, and pair again. ^KA",
    "gs_131": "Make sure both devices have AirDrop set to 'Everyone' or 'Contacts Only' in Control Center. Also check that Wi-Fi and Bluetooth are both on. They need to be within about 30 feet. ^KA",
    "gs_147": "Battery life is important! Check Settings > Battery to see which apps are using the most power. Also try turning off Background App Refresh and reducing screen brightness. DM us for more tips. ^KA",
    "gs_148": "That's definitely not ideal. Let's check a few things: open Activity Monitor (search in Spotlight), click the CPU tab, and let us know what's using the most resources. A restart can also help. ^KA",
    "gs_149": "'Other' storage can build up over time. Try offloading unused apps (Settings > General > iPhone Storage), clearing Safari cache, and restarting. A backup-and-restore can also help clear it. ^KA",
    "gs_151": "At 79%, your battery has degraded somewhat. Apple recommends replacement when it drops below 80%. You can get it replaced at an Apple Store or authorized provider — check https://getsupport.apple.com. ^KA",
    "gs_164": "We're glad you're enjoying it! Thanks for the kind words. If you ever need help with anything, we're always here. Have a great day! 😊 ^KA",
    "gs_166": "We're sorry to hear that was your experience. We'd like to make it right. Please DM us with the details and we'll look into what happened. ^KA",
    "gs_177": "We appreciate you sharing that suggestion! Feature feedback is really valuable. You can submit it directly at apple.com/feedback — the product teams review every submission. ^KA",
    "gs_181": "Hello! 👋 How can we help you today? ^KA",
    "gs_183": "We focus on Apple products and services, but for Samsung support you can reach out to @SamsungSupport. Hope you get the help you need! ^KA",
    "gs_186": "The Apple Store in NYC (Fifth Avenue) is open 24/7! Other locations typically open 10am-9pm. You can check specific hours at apple.com/retail. ^KA",
    "gs_191": "Yes! Apple offers certified refurbished Macs at apple.com/shop/refurbished. They come with a one-year warranty and look and perform like new. ^KA",
}

# Apply reference replies
updated = 0
for example in data:
    if example["id"] in REFERENCE_REPLIES:
        example["reference_reply"] = REFERENCE_REPLIES[example["id"]]
        updated += 1

# Save
with open(config.GOLDEN_SET_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"[OK] Updated {updated} examples with reference replies in {config.GOLDEN_SET_PATH}")
print(f"     Total examples: {len(data)}")
print(f"     Examples with reference_reply: {sum(1 for e in data if e.get('reference_reply', '').strip())}")
