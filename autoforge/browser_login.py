"""Reusable, non-secret browser authentication navigation and status checks."""
from urllib.parse import urlsplit

GMAIL = "https://mail.google.com/mail/u/0/"

def login_step(browser, action="status", url=""):
    """Navigate or inspect. Never assert a login without account evidence.

    QR scanning, Duo and Windows Hello remain with the actual authentication
    provider; no secrets are accepted, logged, captured, or bypassed here.
    """
    if action == "gmail":
        browser.goto(GMAIL, wait=True, timeout=30)
    elif action == "wechat":
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("wechat requires an explicit HTTPS site URL")
        browser.goto(url, wait=True, timeout=30)
    elif action != "status":
        raise ValueError("action must be status, gmail or wechat")
    return browser.evaluate(r"""(() => {
      const host = location.hostname.toLowerCase();
      const text = (document.body?.innerText || '').slice(0, 12000);
      const links = [...document.querySelectorAll('[aria-label],[title]')].map(e => (e.getAttribute('aria-label') || '') + ' ' + (e.getAttribute('title') || '')).join(' ');
      const mail = host === 'mail.google.com';
      const account = (links + ' ' + text).match(/[A-Z0-9._%+-]+@georgetown\.edu/i);
      let status = 'unknown';
      if (mail && account && account[0].toLowerCase() === 'yz1571@georgetown.edu') status = 'gmail_account_visible';
      else if (mail) status = 'gmail_account_unverified';
      else if (host === 'shibb-idp.georgetown.edu') status = 'georgetown_login';
      else if (host === 'duosecurity.com' || host.endsWith('.duosecurity.com')) status = /pin|passkey|security key|安全密钥|通行密钥/i.test(text) ? 'duo_local_auth_prompt' : 'duo_pending';
      else if (/微信|wechat/i.test(text) && (/扫码|扫描|二维码|qr code|scan/i.test(text) || document.querySelector('img[src*=qr],canvas'))) status = 'wechat_qr_possible';
      return {status, origin:location.origin, title:document.title, account:account ? account[0] : null, note:'Status is observational; verify the site login after authentication.'};
    })()""")
