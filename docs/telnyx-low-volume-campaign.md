# Telnyx low-volume campaign checklist

This is the registration manifest for the private JJ Miller & Co LLC lead-review alert program. It
contains no API keys, phone numbers, Telnyx account identifiers, or other credentials.

Do not submit the campaign until every gate below is complete. Telnyx charges a non-refundable
review fee and the initial three-month campaign fee when the campaign is created.

## Pre-submission gates

- The Telnyx brand is verified and its display name is exactly `JJ Miller & Co LLC`.
- `https://jjmillerco.com/legal/` publicly displays the SMS terms and privacy policy.
- `https://approve.jjmillerco.com/health` publicly resolves and returns exactly `ok` while the Mac
  service and tunnel are running.
- The sole recipient has given express written consent using the disclosure under "Consent record"
  below. Keep the dated record locally; do not commit it.
- The sender belongs to the intended Telnyx messaging profile and campaign.
- Telnyx Keyword Management has a HELP auto-response matching the registered help message. Telnyx
  continues to manage its built-in STOP/START behavior and block list.
- The Telnyx balance covers the current review fee and initial three-month LOW_VOLUME fee.

## Campaign request

Replace only `<TELNYX_BRAND_ID>` before submission. Keep the message purpose and wording aligned with
the application. The sample tokens are 43-character non-secret examples matching production token
length.

```json
{
  "brandId": "<TELNYX_BRAND_ID>",
  "usecase": "LOW_VOLUME",
  "subUsecases": [
    "ACCOUNT_NOTIFICATION"
  ],
  "description": "JJ Miller & Co LLC sends low-volume transactional lead-review alerts only to an authorized company administrator. Each alert contains a time-limited link to review a potential lead stored on the company's local Mac. The program sends no marketing or customer messages.",
  "sample1": "JJ Miller & Co LLC lead 95: decks. Review: https://approve.jjmillerco.com/review/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA Reply STOP to opt out.",
  "sample2": "JJ Miller & Co LLC lead. Review: https://approve.jjmillerco.com/review/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB Reply STOP to opt out.",
  "messageFlow": "An authorized JJ Miller & Co LLC administrator provides express written consent during private system setup. Before the number is stored, the administrator accepts this disclosure: 'I agree to receive recurring automated JJ Miller & Co LLC lead-review alerts at the number I provide. Message frequency varies. Message and data rates may apply. Reply STOP to opt out or HELP for help. Consent is not a condition of purchase. Terms and Privacy: https://jjmillerco.com/legal/'. The number is stored in local configuration only after affirmative consent. The program then sends a confirmation message.",
  "optinKeywords": "START",
  "optinMessage": "JJ Miller & Co LLC: You are subscribed to lead-review alerts. Msg frequency varies. Msg & data rates may apply. Reply HELP for help, STOP to opt out.",
  "optoutKeywords": "STOP,STOPALL,UNSUBSCRIBE,CANCEL,END,QUIT",
  "optoutMessage": "JJ Miller & Co LLC: You are unsubscribed from lead-review alerts and will receive no further messages.",
  "helpKeywords": "HELP",
  "helpMessage": "JJ Miller & Co LLC lead-review alerts: Help at jeremy@jjmillerco.com. Reply STOP to opt out.",
  "embeddedLink": true,
  "embeddedLinkSample": "https://approve.jjmillerco.com/review/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
  "embeddedPhone": false,
  "numberPool": false,
  "ageGated": false,
  "directLending": false,
  "subscriberOptin": true,
  "subscriberOptout": true,
  "subscriberHelp": true,
  "termsAndConditions": true,
  "privacyPolicyLink": "https://jjmillerco.com/legal/",
  "termsAndConditionsLink": "https://jjmillerco.com/legal/",
  "autoRenewal": true,
  "referenceId": "jjmiller-lead-review-v1"
}
```

## Consent record

Present this disclosure without alteration and obtain an affirmative written response before campaign
submission:

> I agree to receive recurring automated JJ Miller & Co LLC lead-review alerts at the number I
> provide. Message frequency varies. Message and data rates may apply. Reply STOP to opt out or HELP
> for help. Consent is not a condition of purchase. Terms and Privacy:
> https://jjmillerco.com/legal/

Record the disclosure, affirmative response, recipient number, date, time, and time zone in a local
business record outside Git. Consent applies only to this internal lead-review program.

## After carrier approval

1. Assign the approved campaign to the sender number and messaging profile.
2. Confirm the HELP auto-response and Telnyx STOP/START behavior with the opted-in recipient.
3. Start `lead-agent remote-approval` and verify the public `/health` endpoint.
4. Send one real candidate alert and confirm the review page works over cellular data.
5. Confirm a second notification cycle does not send a duplicate alert.
